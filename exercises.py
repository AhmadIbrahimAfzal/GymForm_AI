class Exercise:
    def __init__(self):
        self.rep_count = 0
        self.stage = "down"

    def update(self, angles, smoothed_class):
        raise NotImplementedError("Each exercise must implement its own update logic")


class BicepCurl(Exercise):
    def __init__(self):
        super().__init__()
        self.stage = "down"
        
    def update(self, angles, smoothed_class):
        le_ang = angles.get('l_elbow', 180)
        
        if le_ang > 150: 
            self.stage = "down"
            
        if le_ang < 50 and self.stage == "down": 
            self.stage = "up"
            # Only count rep if form is good
            if 'Good' in smoothed_class:
                self.rep_count += 1
                
        return self.rep_count, self.stage


class Squat(Exercise):
    def __init__(self):
        super().__init__()
        self.stage = "up"
        
    def update(self, angles, smoothed_class):
        lk_ang = angles.get('l_knee', 180)
        
        if lk_ang > 160:
            self.stage = "up"
            
        if lk_ang < 90 and self.stage == "up":
            self.stage = "down"
            if 'Good' in smoothed_class:
                self.rep_count += 1
                
        return self.rep_count, self.stage
