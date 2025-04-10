import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
from enum import Enum


class GestureMode(Enum):
    IDLE = 0
    MOVING = 1
    LEFT_CLICK = 2
    RIGHT_CLICK = 3
    DRAGGING = 4
    SCROLLING = 5
    DOUBLE_CLICK = 6


class HandGestureMouse:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Screen parameters
        self.screen_width, self.screen_height = pyautogui.size()

        # Mouse control parameters
        self.smoothening = 7
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        self.frame_reduction = 150  # Boundary box reduction
        self.mouse_sensitivity = 1.5  # Higher = more sensitive

        # Gesture parameters
        self.current_mode = GestureMode.IDLE
        self.click_cooldown = 0
        self.double_click_threshold = 0.5  # seconds
        self.last_click_time = 0
        self.drag_start_time = 0
        self.scroll_last_y = 0
        self.scroll_start_y = 0
        self.scroll_threshold = 50  # pixels
        self.dragging = False
        self.handedness = "right"

        # UI parameters
        self.ui_color = (44, 62, 80)  # Dark blue-gray
        self.ui_accent = (52, 152, 219)  # Bright blue
        self.ui_warning = (231, 76, 60)  # Red
        self.ui_success = (46, 204, 113)  # Green
        self.ui_text = (236, 240, 241)  # Light gray
        self.ui_highlight = (241, 196, 15)  # Yellow

        # Performance optimization
        self.last_gesture_check = 0
        self.gesture_check_interval = 0.05  # seconds

    def detect_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Custom drawing style for landmarks
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
        return img, results

    def find_position(self, img, results):
        h, w, c = img.shape
        landmark_list = []

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label.lower()
            self.handedness = handedness

            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])

                # Draw circles on key landmarks with different colors
                if id == 4:  # thumb
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)  # Blue
                elif id == 8:  # index
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)  # Green
                elif id == 12:  # middle
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)  # Red
                elif id == 16:  # ring
                    cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)  # Cyan
                elif id == 20:  # pinky
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)  # Magenta

        return landmark_list

    def count_fingers_up(self, landmark_list):
        if len(landmark_list) == 0:
            return []

        fingers = []

        # Thumb (different logic for left/right hand)
        if self.handedness == "right":
            thumb_up = landmark_list[4][1] < landmark_list[3][1] - 20  # Right hand thumb
        else:
            thumb_up = landmark_list[4][1] > landmark_list[3][1] + 20  # Left hand thumb
        fingers.append(1 if thumb_up else 0)

        # 4 fingers
        for id in [8, 12, 16, 20]:  # Index, middle, ring, pinky tips
            if landmark_list[id][2] < landmark_list[id - 2][2]:  # Tip is above the second joint
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def get_gesture_mode(self, fingers, landmark_list):
        if len(landmark_list) == 0:
            return GestureMode.IDLE

        thumb_x, thumb_y = landmark_list[4][1], landmark_list[4][2]
        index_x, index_y = landmark_list[8][1], landmark_list[8][2]
        middle_x, middle_y = landmark_list[12][1], landmark_list[12][2]
        ring_x, ring_y = landmark_list[16][1], landmark_list[16][2]
        pinky_x, pinky_y = landmark_list[20][1], landmark_list[20][2]

        # Calculate distances
        index_middle_dist = math.hypot(index_x - middle_x, index_y - middle_y)
        thumb_index_dist = math.hypot(thumb_x - index_x, thumb_y - index_y)
        thumb_pinky_dist = math.hypot(thumb_x - pinky_x, thumb_y - pinky_y)

        # Only index finger up - moving mode
        if fingers == [0, 1, 0, 0, 0]:
            return GestureMode.MOVING

        # Index and middle fingers up - clicking mode
        elif fingers == [0, 1, 1, 0, 0] and index_middle_dist < 40:
            current_time = time.time()
            if current_time - self.last_click_time < self.double_click_threshold:
                return GestureMode.DOUBLE_CLICK
            self.last_click_time = current_time
            return GestureMode.LEFT_CLICK

        # Thumb and index finger close - right click
        elif fingers == [1, 1, 0, 0, 0] and thumb_index_dist < 40:
            return GestureMode.RIGHT_CLICK

        # Thumb and index finger extended - dragging mode
        elif fingers == [1, 1, 0, 0, 0] and thumb_index_dist > 60:
            return GestureMode.DRAGGING

        # All fingers up - scrolling mode
        elif fingers == [1, 1, 1, 1, 1]:
            return GestureMode.SCROLLING

        return GestureMode.IDLE

    def process_gesture(self, img, landmark_list):
        if len(landmark_list) == 0:
            self.current_mode = GestureMode.IDLE
            return img

        # Only check gestures at intervals for better performance
        current_time = time.time()
        if current_time - self.last_gesture_check >= self.gesture_check_interval:
            fingers = self.count_fingers_up(landmark_list)
            new_mode = self.get_gesture_mode(fingers, landmark_list)

            # Add hysteresis to prevent mode flickering
            if new_mode != self.current_mode:
                # Special handling for dragging to prevent accidental drag starts
                if new_mode == GestureMode.DRAGGING and self.current_mode != GestureMode.MOVING:
                    new_mode = GestureMode.IDLE

                self.current_mode = new_mode
                self.last_gesture_check = current_time

        index_x, index_y = landmark_list[8][1], landmark_list[8][2]
        thumb_x, thumb_y = landmark_list[4][1], landmark_list[4][2]

        # Handle the current mode
        if self.current_mode == GestureMode.MOVING:
            # Convert coordinates to screen coordinates
            h, w, c = img.shape
            mouse_x = np.interp(index_x, (self.frame_reduction, w - self.frame_reduction),
                                (0, self.screen_width))
            mouse_y = np.interp(index_y, (self.frame_reduction, h - self.frame_reduction),
                                (0, self.screen_height))

            # Smoothen values
            self.curr_x = self.prev_x + (mouse_x - self.prev_x) / self.smoothening
            self.curr_y = self.prev_y + (mouse_y - self.prev_y) / self.smoothening

            # Apply sensitivity
            self.curr_x = self.prev_x + (self.curr_x - self.prev_x) * self.mouse_sensitivity
            self.curr_y = self.prev_y + (self.curr_y - self.prev_y) * self.mouse_sensitivity

            # Move the mouse
            pyautogui.moveTo(self.curr_x, self.curr_y)
            cv2.circle(img, (index_x, index_y), 15, self.ui_success, cv2.FILLED)
            self.prev_x, self.prev_y = self.curr_x, self.curr_y

        elif self.current_mode == GestureMode.LEFT_CLICK:
            pyautogui.click()
            cv2.circle(img, (index_x, index_y), 15, self.ui_warning, cv2.FILLED)
            # Reset to moving mode after click
            self.current_mode = GestureMode.MOVING

        elif self.current_mode == GestureMode.DOUBLE_CLICK:
            pyautogui.doubleClick()
            cv2.circle(img, (index_x, index_y), 15, self.ui_highlight, cv2.FILLED)
            # Reset to moving mode after double click
            self.current_mode = GestureMode.MOVING

        elif self.current_mode == GestureMode.RIGHT_CLICK:
            pyautogui.rightClick()
            cv2.circle(img, (thumb_x, thumb_y), 15, self.ui_warning, cv2.FILLED)
            # Reset to moving mode after right click
            self.current_mode = GestureMode.MOVING

        elif self.current_mode == GestureMode.DRAGGING:
            if not self.dragging:
                pyautogui.mouseDown()
                self.dragging = True
                self.drag_start_time = time.time()

            # Convert coordinates to screen coordinates
            h, w, c = img.shape
            mouse_x = np.interp(index_x, (self.frame_reduction, w - self.frame_reduction),
                                (0, self.screen_width))
            mouse_y = np.interp(index_y, (self.frame_reduction, h - self.frame_reduction),
                                (0, self.screen_height))

            # Smoothen values
            self.curr_x = self.prev_x + (mouse_x - self.prev_x) / self.smoothening
            self.curr_y = self.prev_y + (mouse_y - self.prev_y) / self.smoothening

            # Move the mouse while dragging
            pyautogui.moveTo(self.curr_x, self.curr_y)
            cv2.circle(img, (index_x, index_y), 15, self.ui_highlight, cv2.FILLED)
            self.prev_x, self.prev_y = self.curr_x, self.curr_y

        elif self.current_mode == GestureMode.SCROLLING:
            if not hasattr(self, 'scroll_start_y'):
                self.scroll_start_y = index_y
                self.scroll_last_y = index_y

            scroll_diff = index_y - self.scroll_last_y
            if abs(scroll_diff) > 5:  # Add a small threshold to prevent micro-scrolling
                scroll_amount = -scroll_diff * 2  # Multiply for faster scrolling
                pyautogui.scroll(scroll_amount)
                self.scroll_last_y = index_y

            cv2.circle(img, (index_x, index_y), 15, self.ui_accent, cv2.FILLED)

        else:  # IDLE mode
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
                # After dragging, go back to moving mode
                self.current_mode = GestureMode.MOVING

        return img

    def draw_info(self, img, landmark_list):
        h, w, c = img.shape

        # Draw boundary box with rounded corners
        box_color = self.ui_accent if len(landmark_list) > 0 else self.ui_color
        cv2.rectangle(img, (self.frame_reduction, self.frame_reduction),
                      (w - self.frame_reduction, h - self.frame_reduction), box_color, 2)

        # Draw rounded corners
        corner_radius = 20
        # Top-left
        cv2.ellipse(img, (self.frame_reduction + corner_radius, self.frame_reduction + corner_radius),
                    (corner_radius, corner_radius), 0, 180, 270, box_color, 2)
        # Top-right
        cv2.ellipse(img, (w - self.frame_reduction - corner_radius, self.frame_reduction + corner_radius),
                    (corner_radius, corner_radius), 0, 270, 360, box_color, 2)
        # Bottom-left
        cv2.ellipse(img, (self.frame_reduction + corner_radius, h - self.frame_reduction - corner_radius),
                    (corner_radius, corner_radius), 0, 90, 180, box_color, 2)
        # Bottom-right
        cv2.ellipse(img, (w - self.frame_reduction - corner_radius, h - self.frame_reduction - corner_radius),
                    (corner_radius, corner_radius), 0, 0, 90, box_color, 2)

        # Create a semi-transparent overlay for the info panel
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), self.ui_color, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Draw gesture info
        if len(landmark_list) > 0:
            fingers = self.count_fingers_up(landmark_list)

            # Mode indicator with colored background based on mode
            if self.current_mode == GestureMode.MOVING:
                mode_color = self.ui_success
            elif self.current_mode in [GestureMode.LEFT_CLICK, GestureMode.RIGHT_CLICK, GestureMode.DOUBLE_CLICK]:
                mode_color = self.ui_warning
            elif self.current_mode == GestureMode.DRAGGING:
                mode_color = self.ui_highlight
            elif self.current_mode == GestureMode.SCROLLING:
                mode_color = self.ui_accent
            else:
                mode_color = self.ui_color

            # Draw mode background
            cv2.rectangle(img, (10, 10), (250, 70), mode_color, -1)
            cv2.rectangle(img, (10, 10), (250, 70), self.ui_text, 2)

            # Mode text
            mode_text = f"{self.current_mode.name}"
            cv2.putText(img, mode_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ui_text, 2)

            # Handedness indicator
            hand_text = f"Hand: {self.handedness.capitalize()}"
            cv2.putText(img, hand_text, (270, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui_text, 2)

            # Finger count
            finger_text = f"Fingers: {sum(fingers)}"
            cv2.putText(img, finger_text, (450, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui_text, 2)

            # Instructions at the bottom
            instructions = [
                "Instructions:",
                "1 Finger - Move | 2 Fingers - Click | Thumb+Index - Right Click/Drag | All Fingers - Scroll"
            ]
            for i, line in enumerate(instructions):
                cv2.putText(img, line, (10, h - 30 - (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_text, 1)
        else:
            # No hand detected message
            cv2.putText(img, "No hand detected", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ui_text, 2)

        return img

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video capture")
            return

        # Set PyAutoGUI parameters
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01

        try:
            while True:
                success, img = cap.read()
                if not success:
                    print("Error: Could not read frame")
                    break

                # Flip the image horizontally for a mirror view
                img = cv2.flip(img, 1)

                # Detect hands
                img, results = self.detect_hands(img)

                # Find hand landmarks
                landmark_list = self.find_position(img, results)

                # Process gestures
                img = self.process_gesture(img, landmark_list)

                # Draw information
                img = self.draw_info(img, landmark_list)

                # Display the image
                cv2.imshow("Hand Gesture Mouse Control", img)

                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            # Ensure mouse button is released if program exits during drag
            if self.dragging:
                pyautogui.mouseUp()


if __name__ == "__main__":
    hand_mouse = HandGestureMouse()
    hand_mouse.run()