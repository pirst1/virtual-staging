import torch

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class ImageCaptioner():
    """
    LLaVA-1.6

    Source: https://huggingface.co/liuhaotian/llava-v1.6-34b
    """
    def __init__(self):
        self.processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf"
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
    
    def caption_image(self, image, assistant):
        user_str = f"[INST] <image>\nDescribe the interior image. It must begin with room type description including its style and its color, structure and layout, then describe the all furniture items and their deployment in the layout of the room. [/INST]"
        assistant_str = ""
        for a in assistant:
            assistant_str += "\nASSISTANT: {a}"
        assistant_str = "\nASSISTANT:"
        user_str += assistant_str
        
        inputs = self.processor(
            images=image,
            text=user_str,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].size(1)

        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=77,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][input_len:]
        
        caption = self.processor.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        return caption