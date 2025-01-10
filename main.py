import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pygame
import sys

class ShapeRecognitionGame:
    def __init__(self):
        pygame.init()
        
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Shape Recognition AI Game")
        
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (220, 53, 69)
        self.BLUE = (13, 110, 253)
        self.LIGHT_GRAY = (248, 249, 250)
        self.DARK_GRAY = (33, 37, 41)
        self.BORDER_COLOR = (222, 226, 230)
        
        self.drawing = False
        self.drawing_surface = pygame.Surface((700, 700))
        self.drawing_surface.fill(self.WHITE)
        
        self.scaler = StandardScaler()
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50),
                                 max_iter=500,
                                 random_state=42)
        
        self.X_train = []
        self.y_train = []
        self.shapes = ['circle', 'square', 'triangle']
        self.load_initial_training_data()
        
        self.score = 0
        self.current_shape = None
        self.rounds = 0
        self.feedback_message = ""
        self.feedback_color = self.BLACK
        self.feedback_timer = 0
        
    def load_initial_training_data(self):
        """Load some initial training data for the shapes"""
        for _ in range(10):
            for shape, mean in [('circle', 0.8), ('square', 0.5), ('triangle', 0.3)]:
                features = np.random.normal(mean, 0.1, 64)
                self.X_train.append(features)
                self.y_train.append(shape)
        
        X = np.array(self.X_train)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, self.y_train)
    
    def draw_rounded_rect(self, surface, color, rect, radius=20):
        """Draw a rectangle with rounded corners"""
        x, y, width, height = rect
        
        pygame.draw.rect(surface, color, (x + radius, y, width - 2*radius, height))
        pygame.draw.rect(surface, color, (x, y + radius, width, height - 2*radius))
        
        pygame.draw.circle(surface, color, (x + radius, y + radius), radius)
        pygame.draw.circle(surface, color, (x + width - radius, y + radius), radius)
        pygame.draw.circle(surface, color, (x + radius, y + height - radius), radius)
        pygame.draw.circle(surface, color, (x + width - radius, y + height - radius), radius)
    
    def draw_ui(self):
        """Draw the enhanced game UI"""
        self.screen.fill(self.LIGHT_GRAY)
        
        shadow_offset = 5
        pygame.draw.rect(self.screen, self.BORDER_COLOR, 
                        (45, 45, 710, 710))
        
        pygame.draw.rect(self.screen, self.WHITE, 
                        (40, 40, 710, 710))
        
        self.screen.blit(self.drawing_surface, (40, 40))
        
        panel_x = 800
        
        title_font = pygame.font.Font(None, 60)
        header_font = pygame.font.Font(None, 48)
        regular_font = pygame.font.Font(None, 36)
        
        title = title_font.render('Shape Recognition AI', True, self.DARK_GRAY)
        self.screen.blit(title, (panel_x, 40))
        
        self.draw_rounded_rect(self.screen, self.BLUE, (panel_x, 120, 350, 80))
        score_text = header_font.render(f'Score: {self.score}', True, self.WHITE)
        self.screen.blit(score_text, (panel_x + 20, 140))
        
        if self.feedback_message and self.feedback_timer > 0:
            feedback_text = header_font.render(self.feedback_message, True, self.feedback_color)
            feedback_y = 220   
            self.screen.blit(feedback_text, (panel_x + 20, feedback_y))
            self.feedback_timer -= 1
        
        if self.current_shape:
            shape_header = header_font.render('Draw a:', True, self.DARK_GRAY)
            self.screen.blit(shape_header, (panel_x, 300))
            
            self.draw_rounded_rect(self.screen, self.RED, (panel_x, 350, 350, 80))
            shape_text = header_font.render(self.current_shape.upper(), True, self.WHITE)
            self.screen.blit(shape_text, (panel_x + 20, 370))
        
        self.draw_rounded_rect(self.screen, self.WHITE, (panel_x, 480, 350, 340))
        
        instructions_title = header_font.render('How to Play', True, self.DARK_GRAY)
        self.screen.blit(instructions_title, (panel_x + 20, 500))
        
        instructions = [
            "🖱️ Draw with mouse",
            "⌨️ SPACE to submit",
            "🔄 C to clear canvas",
            "❌ Q to quit",
            "",
            "💡 Tips:",
            "• Draw clearly",
            "• Center your shape",
            "• Be consistent"
        ]
        
        y = 560
        for instruction in instructions:
            text = regular_font.render(instruction, True, self.DARK_GRAY)
            self.screen.blit(text, (panel_x + 20, y))
            y += 35
        
        pygame.display.flip()

    
    def show_feedback(self, message, color):
        """Display feedback message"""
        self.feedback_message = message
        self.feedback_color = color
        self.feedback_timer = 60  
    
    def extract_features(self, surface):
        """Extract features from the drawing"""
        surface_array = pygame.surfarray.array3d(surface)
        gray = cv2.cvtColor(surface_array, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        features = resized.flatten() / 255.0
        return features
    
    def predict_shape(self, features):
        """Predict the drawn shape using the ML model"""
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        return prediction
    
    def update_model(self, features, shape):
        """Update the model with new training data"""
        self.X_train.append(features)
        self.y_train.append(shape)
        
        X = np.array(self.X_train)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, self.y_train)
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        last_pos = None
        
        while True:
            if not self.current_shape:
                self.current_shape = np.random.choice(self.shapes)
                self.rounds += 1
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if 40 <= x <= 750 and 40 <= y <= 750:
                        self.drawing = True
                        last_pos = (x - 40, y - 40)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.drawing = False
                    last_pos = None
                
                elif event.type == pygame.MOUSEMOTION and self.drawing:
                    x, y = event.pos
                    if 40 <= x <= 750 and 40 <= y <= 750:
                        current_pos = (x - 40, y - 40)
                        if last_pos:
                            pygame.draw.line(self.drawing_surface, self.BLACK,
                                          last_pos, current_pos, 4)
                        last_pos = current_pos
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        features = self.extract_features(self.drawing_surface)
                        predicted_shape = self.predict_shape(features)
                        
                        if predicted_shape == self.current_shape:
                            self.score += 1
                            self.show_feedback("Correct!", self.BLUE)
                        else:
                            self.show_feedback(f"Not quite! I saw a {predicted_shape}", self.RED)
                        
                        self.update_model(features, self.current_shape)
                        self.drawing_surface.fill(self.WHITE)
                        self.current_shape = None
                    
                    elif event.key == pygame.K_c:
                        self.drawing_surface.fill(self.WHITE)
                    
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
            
            self.draw_ui()
            clock.tick(60)

if __name__ == "__main__":
    game = ShapeRecognitionGame()
    game.run()