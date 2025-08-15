# virtual_calculator.py
import cv2
import numpy as np
import time
import math
import mediapipe as mp
import ast
import operator

# === Safe evaluator for arithmetic expressions ===
# Allowed operators mapping for AST evaluation
_allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def safe_eval(expr: str):
    """Safely evaluate a numeric arithmetic expression using AST.
    Supports +, -, *, /, %, //, ** and parentheses. Raises ValueError on invalid input.
    """
    if expr.strip() == "":
        return ""
    try:
        node = ast.parse(expr, mode='eval')
    except Exception:
        raise ValueError("Invalid expression")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Num):  # For Python <3.8
            return node.n
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):  # Py3.8+
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _allowed_operators:
                raise ValueError("Operator not allowed")
            left = _eval(node.left)
            right = _eval(node.right)
            return _allowed_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _allowed_operators:
                raise ValueError("Operator not allowed")
            operand = _eval(node.operand)
            return _allowed_operators[op_type](operand)
        else:
            raise ValueError("Disallowed expression")
    return _eval(node)

# === Mediapipe Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Button class ===
class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos  # (x, y)
        self.width = width
        self.height = height
        self.value = value

    def draw(self, frame, hover=False):
        x, y = self.pos
        # hover color vs normal color
        if hover:
            color = (0, 200, 0)       # green-ish when hovered
            border = (0, 150, 0)
        else:
            color = (40, 40, 40)      # dark grey background
            border = (120, 120, 120)
        cv2.rectangle(frame, (x, y), (x + self.width, y + self.height), color, cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + self.width, y + self.height), border, 2)
        # center the text roughly
        text_size = cv2.getTextSize(self.value, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = x + (self.width - text_size[0]) // 2
        text_y = y + (self.height + text_size[1]) // 2 - 6
        cv2.putText(frame, self.value, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    def is_hover(self, x, y):
        bx, by = self.pos
        return bx <= x <= bx + self.width and by <= y <= by + self.height

# === Calculator layout ===
keys = [
    ["7", "8", "9", "+"],
    ["4", "5", "6", "-"],
    ["1", "2", "3", "*"],
    ["C", "0", "=", "/"]
]

button_list = []
btn_w, btn_h = 90, 70
x_offset = 40
y_offset = 140
gap_x = 15
gap_y = 15

for i in range(4):
    for j in range(4):
        xpos = x_offset + j * (btn_w + gap_x)
        ypos = y_offset + i * (btn_h + gap_y)
        button_list.append(Button((xpos, ypos), btn_w, btn_h, keys[i][j]))

expression = ""
last_click_time = 0
click_delay = 0.6  # seconds debounce

# === Start Webcam & Mediapipe Hands ===
cap = cv2.VideoCapture(0)
# Set camera resolution
cap.set(3, 1920)
cap.set(4, 1080)

# Make window full screen
cv2.namedWindow("Virtual Calculator - Day 7", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Virtual Calculator - Day 7", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Draw the display rectangle
        cv2.rectangle(frame, (x_offset, 50), (x_offset + 4 * (btn_w + gap_x) - gap_x, 120), (20, 20, 20), cv2.FILLED)
        # Expression text (right-aligned)
        disp_x = x_offset + 10
        disp_y = 110
        cv2.putText(frame, expression, (disp_x, disp_y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 200), 2)

        # Default: no hovered button
        hover_btn = None
        pinch_cx = pinch_cy = None
        # Process hand landmarks
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if lm_list:
                # thumb tip = 4, index tip = 8
                x1, y1 = lm_list[4]
                x2, y2 = lm_list[8]
                pinch_cx, pinch_cy = (x1 + x2) // 2, (y1 + y2) // 2
                # Distance between thumb tip and index tip
                length = math.hypot(x2 - x1, y2 - y1)

                # Show a small circle at center between tips
                cv2.circle(frame, (pinch_cx, pinch_cy), 8, (0, 255, 255), cv2.FILLED)

                # Use relative threshold depending on frame size
                threshold = max(30, int(min(w, h) * 0.03))  # dynamic threshold
                is_pinch = length < threshold

                # If pinched, check buttons for click (with debounce)
                if is_pinch and (time.time() - last_click_time) > click_delay:
                    for button in button_list:
                        if button.is_hover(pinch_cx, pinch_cy):
                            val = button.value
                            if val == "C":
                                expression = ""
                            elif val == "=":
                                try:
                                    result_val = safe_eval(expression)
                                    expression = str(result_val)
                                except Exception:
                                    expression = "Error"
                            else:
                                # prevent consecutive operators at start or after another operator (basic cleanup)
                                if expression == "" and val in "+-*/":
                                    pass  # ignore leading operator
                                else:
                                    expression += val
                            last_click_time = time.time()
                            # visual click feedback (bigger circle)
                            cv2.circle(frame, (pinch_cx, pinch_cy), 18, (0, 255, 0), 3)
                            break

        # Draw buttons and hover highlight
        for button in button_list:
            if pinch_cx is not None and button.is_hover(pinch_cx, pinch_cy):
                button.draw(frame, hover=True)
            else:
                button.draw(frame, hover=False)

        cv2.putText(frame, "Press 'q' to quit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("Virtual Calculator - Hand Gestures", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
