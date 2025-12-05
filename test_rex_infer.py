from rex_omni import RexOmniWrapper

# 1. Initialise model with a safe config for your GPU/CPU
wrapper = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",
    backend="transformers",
    # Force safe attention & dtype, avoids flash-attn + bf16 issues
    attn_implementation="eager",
    torch_dtype="float16",   # will still fall back on CPU if needed
    device_map="auto",
)

# 2. Point to the same image your Flask app uploads
image_path = "tutorials/ocr_example/test_images/ocr.png"  # or whatever name you used

result = wrapper.inference(image_path , "OCR")  # or ["detection"] depending on what you want


print(result)