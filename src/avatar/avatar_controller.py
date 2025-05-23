from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os # For saving test images

class AvatarController:
    def __init__(self):
        self.avatar_state = "idle"
        self.current_expression = "neutral"
        self.speaking = False
        print("AvatarController initialized.")
        
    def create_nhs_avatar(self, expression="neutral", speaking=False, width=300, height=400):
        img = Image.new('RGB', (width, height), color='#f0f8ff') # AliceBlue background
        draw = ImageDraw.Draw(img)
        
        # Face
        face_color = '#fdbcb4' # Light skin tone
        draw.ellipse([width*0.25, height*0.2, width*0.75, height*0.55], fill=face_color, outline='#333')
        
        # Hair (simple cap style)
        hair_color = '#3b2219' # Dark brown
        draw.ellipse([width*0.23, height*0.15, width*0.77, height*0.4], fill=hair_color)
        # Redraw face outline over hair for cleaner look
        draw.ellipse([width*0.25, height*0.2, width*0.75, height*0.55], fill=face_color, outline='#333')

        # Eyes
        eye_color = '#4169E1' # Royal Blue
        eye_y = height*0.325
        # Left eye
        draw.ellipse([width*0.33, eye_y, width*0.4, eye_y + height*0.0375], fill='white')
        draw.ellipse([width*0.35, eye_y + height*0.005, width*0.383, eye_y + height*0.0275], fill=eye_color) # Pupil
        # Right eye
        draw.ellipse([width*0.6, eye_y, width*0.67, eye_y + height*0.0375], fill='white')
        draw.ellipse([width*0.616, eye_y + height*0.005, width*0.65, eye_y + height*0.0275], fill=eye_color) # Pupil
        
        # Nose (simple line)
        draw.line([width*0.5, height*0.4, width*0.5, height*0.45], fill='#d29d94', width=int(width*0.02))
        
        # Mouth
        mouth_y_start = height*0.48
        mouth_y_end = height*0.52
        if speaking:
            draw.ellipse([width*0.45, mouth_y_start - height*0.01, width*0.55, mouth_y_end], fill='#333') # Open mouth
        else:
            if expression == "smile":
                draw.arc([width*0.43, mouth_y_start - height*0.01, width*0.57, mouth_y_end], 0, 180, fill='#333', width=int(width*0.01))
            else: # neutral, concerned, friendly (simple line for these)
                draw.line([width*0.45, mouth_y_start + height*0.01, width*0.55, mouth_y_start + height*0.01], fill='#333', width=int(width*0.007))
        
        # Uniform (NHS Blue)
        uniform_color = '#003087' # NHS Blue
        draw.rectangle([width*0.16, height*0.625, width*0.84, height], fill=uniform_color)
        
        # NHS Badge
        draw.rectangle([width*0.4, height*0.7, width*0.6, height*0.8], fill='white', outline='#333')
        try:
            font_size = int(min(width, height) * 0.04)
            try:
                font = ImageFont.truetype("arial.ttf", font_size) 
            except IOError:
                # print("Arial font not found, using default font.")
                font = ImageFont.load_default() 
            draw.text((width*0.45, height*0.72), "NHS", fill=uniform_color, font=font)
        except Exception as e: # Catch any font loading errors
             print(f"Font loading error: {e}. Drawing simple text.")
             draw.text((width*0.45, height*0.72), "NHS", fill=uniform_color) # Fallback
        return img
    
    def get_avatar_base64(self, expression="neutral", speaking=False):
        img = self.create_nhs_avatar(expression, speaking)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"
    
    def animate_speaking(self, expression="neutral", num_frames=4):
        frames = []
        for i in range(num_frames):
            speaking_state = (i % 2 == 0) # Alternate between speaking and not speaking
            img_base64 = self.get_avatar_base64(expression, speaking_state)
            frames.append(img_base64)
        return frames
    
    def set_expression(self, expression: str):
        valid_expressions = ["neutral", "smile", "concerned", "friendly"]
        if expression in valid_expressions:
            self.current_expression = expression
        else:
            self.current_expression = "neutral"
        print(f"Avatar expression set to: {self.current_expression}")
        
    def set_speaking(self, speaking: bool):
        self.speaking = speaking
        print(f"Avatar speaking state set to: {self.speaking}")

class NHSAvatarController(AvatarController):
    def __init__(self):
        super().__init__()
        self.nhs_expressions = {
            "welcome": "smile", "listening": "neutral", "thinking": "neutral",
            "explaining": "friendly", "concerned": "concerned"
        }
        print("NHSAvatarController initialized.")
    
    def get_nhs_avatar(self, context="welcome", speaking_override=None):
        expression = self.nhs_expressions.get(context, "neutral")
        current_speaking_state = self.speaking
        if speaking_override is not None:
            current_speaking_state = speaking_override
        return self.get_avatar_base64(expression, current_speaking_state)

    def create_professional_avatar(self, width=400, height=500):
        # Use the existing create_nhs_avatar and add text or minor adjustments
        img = self.create_nhs_avatar(expression="friendly", speaking=False, width=width, height=height)
        draw = ImageDraw.Draw(img)
        try:
            # Add a name tag or title, slightly larger font for "professional" look
            font_size = int(min(width, height) * 0.03)
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        draw.text((width*0.4, height*0.85), "Sarah - NHS", fill='#003087', font=font, align="center")
        print("Professional avatar created (simulated enhancement).")
        return img

def test_avatar_creation():
    print("\nTesting Avatar Creation (Pillow)...")
    controller = NHSAvatarController()
    assets_images_dir = "assets/images" # This directory should exist from Phase 1
    if not os.path.exists(assets_images_dir):
        os.makedirs(assets_images_dir)
        print(f"Created directory: {assets_images_dir}")
    try:
        # Basic avatar
        img_basic = controller.create_nhs_avatar()
        basic_path = os.path.join(assets_images_dir, "test_avatar_basic.png")
        img_basic.save(basic_path)
        print(f"✅ Basic avatar saved to {basic_path}")
        
        # Professional avatar
        img_prof = controller.create_professional_avatar()
        prof_path = os.path.join(assets_images_dir, "test_avatar_professional.png")
        img_prof.save(prof_path)
        print(f"✅ Professional avatar saved to {prof_path}")
        
        # Base64 conversion test
        base64_img = controller.get_avatar_base64("smile", True)
        if base64_img.startswith("data:image/png;base64,") and len(base64_img) > 50: # Basic check
             print(f"✅ Base64 conversion length OK: {len(base64_img)}")
        else:
             print(f"❌ Base64 conversion failed or unexpected format.")

    except Exception as e:
        print(f"❌ Error during avatar test image saving: {e}")

if __name__ == "__main__":
    pass
    # test_avatar_creation() # Can be uncommented for direct testing if needed outside of main script execution context
