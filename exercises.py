class Exercise:
    def __init__(self):
        self.rep_count = 0
        self.stage = "down"
        self.form_was_bad = False

    def update(self, angles, smoothed_class):
        raise NotImplementedError("Each exercise must implement its own update logic")


class BicepCurl(Exercise):
    def __init__(self):
        super().__init__()
        self.stage = "down"
        
    def update(self, angles, smoothed_class):
        le_ang = angles.get('l_elbow', 180)
        re_ang = angles.get('r_elbow', 180)
        
        # Track form throughout the movement
        if 'Bad' in smoothed_class:
            self.form_was_bad = True
            
        if le_ang > 120 and re_ang > 120: 
            self.stage = "down"
            self.form_was_bad = False # Reset flag when resting at the bottom
            
        if le_ang < 90 and re_ang < 90 and self.stage == "down": 
            self.stage = "up"
            # strict reps only, ensuring form was good the whole way up
            if not self.form_was_bad and 'Good' in smoothed_class:
                self.rep_count += 1
                
        return self.rep_count, self.stage


class Squat(Exercise):
    def __init__(self):
        super().__init__()
        self.stage = "up"
        
    def update(self, angles, smoothed_class):
        lk_ang = angles.get('l_knee', 180)
        rk_ang = angles.get('r_knee', 180)
        
        if 'Bad' in smoothed_class:
            self.form_was_bad = True
            
        if lk_ang > 140 and rk_ang > 140:
            self.stage = "up"
            self.form_was_bad = False # Reset flag when resting at the top
            
        if lk_ang < 115 and rk_ang < 115 and self.stage == "up":
            self.stage = "down"
            if not self.form_was_bad and 'Good' in smoothed_class:
                self.rep_count += 1
                
        return self.rep_count, self.stage


class LateralRaise(Exercise):
    def __init__(self):
        super().__init__()
        self.stage = "down"
        
    def update(self, angles, smoothed_class):
        ls_ang = angles.get('l_shoulder', 0)
        rs_ang = angles.get('r_shoulder', 0)
        
        if 'Bad' in smoothed_class:
            self.form_was_bad = True
            
        if ls_ang > 65 and rs_ang > 65:
            self.stage = "up"
            
        if ls_ang < 45 and rs_ang < 45:
            if self.stage == "up":
                self.stage = "down"
                if not self.form_was_bad and 'Good' in smoothed_class:
                    self.rep_count += 1
            else:
                self.stage = "down"
                self.form_was_bad = False # Reset flag when resting at the bottom
                
        return self.rep_count, self.stage


class ShoulderPress(Exercise):
    def __init__(self):
        super().__init__()
        self.stage = "down"
        
    def update(self, angles, smoothed_class):
        le_ang = angles.get('l_elbow', 180)
        re_ang = angles.get('r_elbow', 180)
        
        if 'Bad' in smoothed_class:
            self.form_was_bad = True
            
        if le_ang < 100 and re_ang < 100:
            self.stage = "down"
            self.form_was_bad = False
            
        if le_ang > 150 and re_ang > 150 and self.stage == "down":
            self.stage = "up"
            if not self.form_was_bad and 'Good' in smoothed_class:
                self.rep_count += 1
                
        return self.rep_count, self.stage


class TricepFinisher(Exercise):
    def __init__(self):
        super().__init__()
        self.stage = "bent"
        
    def update(self, angles, smoothed_class):
        active_arm = angles.get('active_arm', 'left')
        if active_arm == 'left':
            el_ang = angles.get('l_elbow', 180)
        else:
            el_ang = angles.get('r_elbow', 180)
            
        if 'Bad' in smoothed_class:
            self.form_was_bad = True
            
        if el_ang < 70:
            self.stage = "bent"
            self.form_was_bad = False
            
        if el_ang > 150 and self.stage == "bent":
            self.stage = "straight"
            if not self.form_was_bad and 'Good' in smoothed_class:
                self.rep_count += 1
                
        return self.rep_count, self.stage
