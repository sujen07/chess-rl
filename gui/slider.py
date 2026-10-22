import pygame

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val):
        self.rect = pygame.Rect(x, y, w, h)  # Slider rectangle
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val  # Current value
        self.active = False

    def draw(self, screen, font):
        # Draw the slider
        pygame.draw.rect(screen, pygame.Color("grey"), self.rect, 2)
        button_x = self.rect.x + (self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        pygame.draw.circle(screen, pygame.Color("blue"), (int(button_x), self.rect.centery), 10)

        # Draw the text above the slider
        text = font.render(f"AI Skill Level: {int(self.val)}", True, pygame.Color("black"))
        screen.blit(text, (self.rect.x, self.rect.y - 30))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.active = False
        elif event.type == pygame.MOUSEMOTION:
            if self.active:
                # Update the value based on mouse position
                x, _ = event.pos
                relative_x = min(max(x, self.rect.x), self.rect.x + self.rect.width)
                self.val = self.min_val + (relative_x - self.rect.x) / self.rect.width * (self.max_val - self.min_val)
