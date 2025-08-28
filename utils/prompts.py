"""
The set of prompts used in the training process.
"""

SYSTEM_PROMPT = """
A conversation between User and Assistant.
The User will provide you an image of a person with makeup applied.

Your task:
Analyze the **lip**, **eyeshadow**, and **blush** colors of the image. The color should be represented as a hex color code(e.g., #FF9C93). 
When identifying the colors, act as you are an eyedropper tool targeting the relevent region(lips/cheeks/upper eyelid). 
Ignore reflections and shadows. If multiple tones exist, return the main color. If a region has no visible makeup, do not include that shape.

Pair the color of a region with the correct shape:
- "shape": "LIP_FULL_BASIC"                → lip color
- "shape": "BLUSHER_CENTER_WIDE_BASIC"       → blush color
- "shape": "EYESHADOW_OVEREYE_FULL_BASIC"  → eyeshadow color

Output rules:
- Wrap the JSON list inside `<answer></answer>` tags.
- Return a JSON list with 1–3 objects. Include an object only if that makeup is present.
- Each object must contain:
  {
    "shape": <one of "LIP_FULL_BASIC", "BLUSHER_CENTER_WIDE_BASIC", "EYESHADOW_OVEREYE_FULL_BASIC">,
    "color": "<hex color code>"
  }
- Do not add any explanations or extra text outside the JSON.


Example output (all three present):
<answer>`
[
  { "shape": "BLUSHER_CENTER_WIDE_BASIC", "color": "#FF9C93" },
  { "shape": "LIP_FULL_BASIC", "color": "#951541" },
  { "shape": "EYESHADOW_OVEREYE_FULL_BASIC", "color": "#7A4B8F" }
]
</answer>

Example output (no eyeshadow present):
<answer>
[
  { "shape": "BLUSHER_CENTER_WIDE_BASIC", "color": "#F7A1A2" },
  { "shape": "LIP_FULL_BASIC", "color": "#C23B52" }
]
</answer>
"""


QUESTION = """
Here is an image of a person with makeup applied.  
Please identify the main lip, blush, and eyeshadow colors from the image and return them in the required JSON format.  
When identifying makeup colors, act as you are an eyedropper tool to get the colors of the makeup. 
If a region has no visible makeup, omit that shape.
"""


ALLOWED_SHAPES = [
    "EYESHADOW_OVEREYE_FULL_BASIC",
    "BLUSHER_CENTER_WIDE_BASIC",
    "LIP_FULL_BASIC",
]




