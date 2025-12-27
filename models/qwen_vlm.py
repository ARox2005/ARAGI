"""
Qwen Vision-Language Model integration module.
Supports text and image understanding with GPU/CPU fallback.
"""

import os
from typing import List, Optional, Dict, Any, Generator, Union
from io import BytesIO
import base64

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import TextIteratorStreamer
from threading import Thread


class QwenVLM:
    """
    Qwen Vision-Language Model for multimodal generation.
    Handles text and image inputs with streaming support.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Qwen VLM model.
        
        Args:
            model_name: HuggingFace model identifier for Qwen VL
            device: Device to use ('cuda', 'cpu', or None for auto)
            torch_dtype: Torch dtype (None for auto selection)
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Auto-select dtype based on device
        if torch_dtype is None:
            if self.device == 'cuda':
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype
        
        print(f"Loading Qwen VLM model '{model_name}' on {self.device}...")
        print(f"This may take a few minutes on first run...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Try to load as VL model, fallback to standard if needed
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.has_vision = True
        except Exception:
            self.processor = None
            self.has_vision = False
            print("Note: Vision processor not available, using text-only mode")
        
        # Load model with appropriate settings
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
            "cache_dir": cache_dir,
        }
        
        # For CUDA, use device_map for automatic distribution
        if self.device == 'cuda':
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Qwen VLM model loaded successfully!")
    
    def _prepare_image(self, image: Union[str, bytes, Image.Image]) -> Image.Image:
        """
        Prepare image for model input.
        
        Args:
            image: Image path, bytes, or PIL Image
            
        Returns:
            PIL Image object
        """
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        elif isinstance(image, bytes):
            return Image.open(BytesIO(image)).convert('RGB')
        elif isinstance(image, str):
            if os.path.exists(image):
                return Image.open(image).convert('RGB')
            elif image.startswith('data:image'):
                # Base64 data URL
                base64_data = image.split(',')[1]
                image_bytes = base64.b64decode(base64_data)
                return Image.open(BytesIO(image_bytes)).convert('RGB')
            else:
                raise ValueError(f"Invalid image path: {image}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _build_messages(
        self,
        query: str,
        context: Optional[str] = None,
        images: Optional[List[Union[str, bytes, Image.Image]]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Build chat messages for the model.
        
        Args:
            query: User query
            context: Optional RAG context
            images: Optional list of images
            system_prompt: Optional system prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # System message
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        else:
            messages.append({
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that ONLY answers questions based on the provided context. "
                    "STRICT RULES:\n"
                    "1. You must ONLY use information from the provided context to answer questions.\n"
                    "2. If the context does not contain relevant information to answer the question, "
                    "you MUST respond with: 'I don't have information about this topic in my knowledge base. "
                    "Please upload relevant documents or ask a question related to the uploaded content.'\n"
                    "3. Do NOT use any external knowledge, general knowledge, or make assumptions beyond the context.\n"
                    "4. Always cite the source documents when providing information from the context.\n"
                    "5. If the question is partially answerable, only answer the parts that are covered by the context "
                    "and clearly state what information is missing."
                )
            })
        
        # Build user message content
        user_content = []
        
        # Add images if present and vision is available
        if images and self.has_vision:
            for img in images:
                pil_image = self._prepare_image(img)
                user_content.append({
                    "type": "image",
                    "image": pil_image
                })
        
        # Add context if present - with strict enforcement
        text_content = ""
        if context and context.strip():
            text_content += f"KNOWLEDGE BASE CONTEXT (Use ONLY this information to answer):\n{context}\n\n"
            text_content += f"Question: {query}\n\nRemember: Answer ONLY using the above context. If the context doesn't contain the answer, say you don't have that information."
        else:
            # No context available - explicitly tell model to refuse
            text_content += (
                "IMPORTANT: No relevant information was found in the knowledge base for this query.\n\n"
                f"Question: {query}\n\n"
                "Since there is NO CONTEXT from the knowledge base, you MUST respond with: "
                "'I don't have information about this topic in my knowledge base. "
                "Please upload relevant documents or ask a question related to the uploaded content.'"
            )
        
        user_content.append({
            "type": "text",
            "text": text_content
        })
        
        messages.append({
            "role": "user",
            "content": user_content if len(user_content) > 1 else text_content
        })
        
        return messages
    
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        images: Optional[List[Union[str, bytes, Image.Image]]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate response for a query.
        
        Args:
            query: User query
            context: Optional RAG context with retrieved documents
            images: Optional list of images to analyze
            system_prompt: Optional custom system prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response text
        """
        messages = self._build_messages(query, context, images, system_prompt)
        
        # Prepare input for Qwen2-VL
        if self.has_vision and self.processor and images:
            # Process with vision
            pil_images = [self._prepare_image(img) for img in images]
            
            # Format text for processing
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=text,
                images=pil_images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            # Text-only processing
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_stream(
        self,
        query: str,
        context: Optional[str] = None,
        images: Optional[List[Union[str, bytes, Image.Image]]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """
        Generate response with streaming output.
        
        Args:
            query: User query
            context: Optional RAG context
            images: Optional images
            system_prompt: Optional system prompt
            max_new_tokens: Max tokens
            temperature: Sampling temperature
            top_p: Top-p parameter
            
        Yields:
            Generated tokens one at a time
        """
        messages = self._build_messages(query, context, images, system_prompt)
        
        # Prepare inputs
        if self.has_vision and self.processor and images:
            pil_images = [self._prepare_image(img) for img in images]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=text,
                images=pil_images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Create streamer with timeout for real-time streaming
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=10.0  # Timeout to yield tokens immediately
        )
        
        # Generation kwargs
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        # Run generation in thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens
        for token in streamer:
            yield token
        
        thread.join()
    
    def is_vision_available(self) -> bool:
        """Check if vision capabilities are available."""
        return self.has_vision


# Fallback model for when full VL model cannot be loaded
class QwenTextModel:
    """
    Fallback text-only Qwen model for systems without sufficient resources.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize text-only Qwen model."""
        self.model_name = model_name
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        
        print(f"Loading Qwen text model '{model_name}' on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
            "cache_dir": cache_dir,
        }
        
        if self.device == 'cuda':
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Text model loaded successfully!")
    
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        images: Optional[List] = None,  # Ignored
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text response (images are ignored)."""
        
        strict_prompt = (
            "You are a helpful AI assistant that ONLY answers questions based on the provided context. "
            "STRICT RULES:\n"
            "1. You must ONLY use information from the provided context to answer questions.\n"
            "2. If the context does not contain relevant information to answer the question, "
            "you MUST respond with: 'I don't have information about this topic in my knowledge base. "
            "Please upload relevant documents or ask a question related to the uploaded content.'\n"
            "3. Do NOT use any external knowledge, general knowledge, or make assumptions beyond the context.\n"
            "4. Always cite the source documents when providing information from the context.\n"
            "5. If the question is partially answerable, only answer the parts that are covered by the context "
            "and clearly state what information is missing."
        )
        
        messages = [
            {
                "role": "system",
                "content": system_prompt or strict_prompt
            }
        ]
        
        user_message = ""
        if context and context.strip():
            user_message += f"KNOWLEDGE BASE CONTEXT (Use ONLY this information to answer):\n{context}\n\n"
            user_message += f"Question: {query}\n\nRemember: Answer ONLY using the above context. If the context doesn't contain the answer, say you don't have that information."
        else:
            # No context available - explicitly tell model to refuse
            user_message += (
                "IMPORTANT: No relevant information was found in the knowledge base for this query.\n\n"
                f"Question: {query}\n\n"
                "Since there is NO CONTEXT from the knowledge base, you MUST respond with: "
                "'I don't have information about this topic in my knowledge base. "
                "Please upload relevant documents or ask a question related to the uploaded content.'"
            )
        
        messages.append({"role": "user", "content": user_message})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_stream(
        self,
        query: str,
        context: Optional[str] = None,
        images: Optional[List] = None,  # Ignored
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Generate text response with streaming output (images are ignored)."""
        
        strict_prompt = (
            "You are a helpful AI assistant that ONLY answers questions based on the provided context. "
            "STRICT RULES:\n"
            "1. You must ONLY use information from the provided context to answer questions.\n"
            "2. If the context does not contain relevant information to answer the question, "
            "you MUST respond with: 'I don't have information about this topic in my knowledge base. "
            "Please upload relevant documents or ask a question related to the uploaded content.'\n"
            "3. Do NOT use any external knowledge, general knowledge, or make assumptions beyond the context.\n"
            "4. Always cite the source documents when providing information from the context.\n"
            "5. If the question is partially answerable, only answer the parts that are covered by the context "
            "and clearly state what information is missing."
        )
        
        messages = [
            {
                "role": "system",
                "content": system_prompt or strict_prompt
            }
        ]
        
        user_message = ""
        if context and context.strip():
            user_message += f"KNOWLEDGE BASE CONTEXT (Use ONLY this information to answer):\n{context}\n\n"
            user_message += f"Question: {query}\n\nRemember: Answer ONLY using the above context. If the context doesn't contain the answer, say you don't have that information."
        else:
            # No context available - explicitly tell model to refuse
            user_message += (
                "IMPORTANT: No relevant information was found in the knowledge base for this query.\n\n"
                f"Question: {query}\n\n"
                "Since there is NO CONTEXT from the knowledge base, you MUST respond with: "
                "'I don't have information about this topic in my knowledge base. "
                "Please upload relevant documents or ask a question related to the uploaded content.'"
            )
        
        messages.append({"role": "user", "content": user_message})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=10.0
        )
        
        # Generation kwargs
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        # Run generation in thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens
        for token in streamer:
            yield token
        
        thread.join()
    
    def is_vision_available(self) -> bool:
        return False


def load_qwen_model(
    prefer_vision: bool = True,
    cache_dir: Optional[str] = None
) -> Union[QwenVLM, QwenTextModel]:
    """
    Load the best available Qwen model based on system capabilities.
    
    Args:
        prefer_vision: Whether to prefer vision model
        cache_dir: Model cache directory
        
    Returns:
        QwenVLM or QwenTextModel instance
    """
    if prefer_vision:
        try:
            return QwenVLM(cache_dir=cache_dir)
        except Exception as e:
            print(f"Could not load vision model: {e}")
            print("Falling back to text-only model...")
    
    return QwenTextModel(cache_dir=cache_dir)
