"""
MediaPipe Hand Tracking Application
Tracks hands in real-time using webcam and displays landmarks with connections.
Press 'q' to quit, 's' to save screenshot, 'h' to toggle help overlay.
"""

import cv2
import mediapipe as mp
import time
import os

class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """Initialize hand tracker with MediaPipe"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_hands=max_hands,
            static_image_mode=False
        )
        
        self.fps = 0
        self.prev_time = 0
        self.show_help = True
        self.fps_history = []
        self.fps_window = 10
        self.process_time = 0
        self.capture_time = 0
        
    def calculate_fps(self):
        """Calculate frames per second with smoothing"""
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time
        
        # Smooth FPS with moving average
        self.fps_history.append(fps)
        if len(self.fps_history) > self.fps_window:
            self.fps_history.pop(0)
        
        self.fps = sum(self.fps_history) / len(self.fps_history)
        return self.fps
    
    def draw_info_overlay(self, frame, hand_count):
        """Draw information overlay on frame"""
        height, width = frame.shape[:2]
        
        # Semi-transparent background for info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display information
        cv2.putText(frame, f'Hands Detected: {hand_count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {int(self.fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Capture: {int(self.capture_time * 1000)}ms', (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 1)
        cv2.putText(frame, f'Process: {int(self.process_time * 1000)}ms', (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 1)
        
        # Help overlay
        if self.show_help:
            help_y = height - 120
            cv2.rectangle(frame, (10, help_y - 10), (350, height - 10), (0, 0, 0), -1)
            cv2.putText(frame, "Controls:", (20, help_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Q - Quit  |  S - Screenshot  |  H - Toggle Help", 
                        (20, help_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def get_finger_positions(self, hand_landmarks):
        """Get fingertip positions (landmarks 4, 8, 12, 16, 20)"""
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        positions = []
        
        for tip_id in finger_tips:
            landmark = hand_landmarks.landmark[tip_id]
            positions.append((landmark.x, landmark.y, landmark.z))
        
        return positions
    
    def process_frame(self, frame):
        """Process frame and detect hands"""
        process_start = time.time()
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        hand_count = 0
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks and connections
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get hand label (Left/Right)
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                    
                    # Get wrist position for label
                    wrist = hand_landmarks.landmark[0]
                    h, w, _ = frame.shape
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    
                    # Draw hand label
                    cv2.putText(frame, handedness, (cx - 30, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Store process time
        self.process_time = time.time() - process_start
        
        # Calculate FPS
        self.calculate_fps()
        
        # Draw overlay
        self.draw_info_overlay(frame, hand_count)
        
        return frame
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"hand_tracking_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        return filename
    
    def run(self, camera_id=0):
        """Run the hand tracking application"""
        cap = cv2.VideoCapture(camera_id)
        
        # Try to set optimal camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Print camera info
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {width}x{height} @ {actual_fps} FPS (requested)")
        print(f"Model Complexity: 0 (fastest)")
        
        print("\nHand Tracking Started!")
        print("Press 'Q' to quit, 'S' to save screenshot, 'H' to toggle help")
        print("\nWatch the timing diagnostics:")
        print("- Capture time = time to grab frame from camera")
        print("- Process time = MediaPipe processing time\n")
        
        while cap.isOpened():
            capture_start = time.time()
            success, frame = cap.read()
            self.capture_time = time.time() - capture_start
            
            if not success:
                print("Failed to capture frame")
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the result
            cv2.imshow('MediaPipe Hand Tracking', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("Exiting...")
                break
            elif key == ord('s') or key == ord('S'):
                self.save_screenshot(processed_frame)
            elif key == ord('h') or key == ord('H'):
                self.show_help = not self.show_help
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Hand tracking stopped.")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    """Main function to run hand tracker"""
    # Create hand tracker with custom settings
    tracker = HandTracker(
        max_hands=2,
        detection_confidence=0.5,
        tracking_confidence=0.5
    )
    
    # Run the tracker
    tracker.run(camera_id=0)


if __name__ == "__main__":
    main()
