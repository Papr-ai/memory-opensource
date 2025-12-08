class ContextWindow:
    def __init__(self, size):
        self.size = size
        self.window = []

    def add(self, message):
        if len(self.window) >= self.size:
            self.window.pop(0)
        self.window.append(message)

    def get(self):
        return self.window
