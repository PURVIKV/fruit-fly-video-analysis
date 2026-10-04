import cv2


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "input/videos/test1.mp4"


# ============================================================
# STEP 1: READ FIRST FRAME
# ============================================================

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return None

    ret, frame = cap.read()

    cap.release()

    if not ret:
        print("ERROR: Could not read the first frame.")
        return None

    return frame


# ============================================================
# STEP 2: PREPROCESS FRAME
# ============================================================

def preprocess_frame(frame):

    # Convert BGR image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce small amounts of noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Automatically find a threshold using Otsu's method
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return gray, binary


# ============================================================
# STEP 3: FIND CONTOURS
# ============================================================

def find_contours(binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


# ============================================================
# STEP 4: EXTRACT AND FILTER CONTOUR FEATURES
# ============================================================

def extract_features(contours, frame_width, frame_height):

    features = []

    for i, contour in enumerate(contours):

        # Calculate contour area
        area = cv2.contourArea(contour)

        # ----------------------------------------------------
        # Remove very small regions
        # ----------------------------------------------------

        if area < 500:
            continue

        # ----------------------------------------------------
        # Calculate bounding rectangle
        # ----------------------------------------------------

        x, y, width, height = cv2.boundingRect(contour)

        # ----------------------------------------------------
        # Remove contours touching the image border
        #
        # The previous output showed a 1530 x 1530 contour.
        # That was the background.
        # ----------------------------------------------------

        touches_border = (
            x <= 0
            or y <= 0
            or x + width >= frame_width
            or y + height >= frame_height
        )

        if touches_border:
            continue

        # Avoid division by zero
        if height == 0:
            continue

        # ----------------------------------------------------
        # Calculate aspect ratio
        # ----------------------------------------------------

        aspect_ratio = width / height

        # ----------------------------------------------------
        # Calculate centroid
        # ----------------------------------------------------

        moments = cv2.moments(contour)

        if moments["m00"] != 0:

            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]

        else:

            center_x = 0
            center_y = 0

        # ----------------------------------------------------
        # Store features
        # ----------------------------------------------------

        features.append({
            "contour_id": i,
            "area": area,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "center_x": center_x,
            "center_y": center_y
        })

    return features


# ============================================================
# STEP 5: DRAW CANDIDATE OBJECTS
# ============================================================

def draw_candidates(frame, contours, features):

    result = frame.copy()

    for item in features:

        contour_id = item["contour_id"]

        x = item["x"]
        y = item["y"]
        width = item["width"]
        height = item["height"]

        center_x = int(item["center_x"])
        center_y = int(item["center_y"])

        area = item["area"]

        # Get the original contour
        contour = contours[contour_id]

        # Draw contour
        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # Draw bounding box
        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2
        )

        # Draw centroid
        cv2.circle(
            result,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        # Label
        label = f"ID {contour_id} | Area {area:.0f}"

        cv2.putText(
            result,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    return result


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("FRUIT FLY IMAGE PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Read frame
    # --------------------------------------------------------

    frame = get_first_frame(VIDEO_PATH)

    if frame is None:
        return

    frame_height, frame_width = frame.shape[:2]

    print("\nFrame information:")
    print("Width :", frame_width)
    print("Height:", frame_height)

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    gray, binary = preprocess_frame(frame)

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours = find_contours(binary)

    print("\nTotal contours detected:", len(contours))

    # --------------------------------------------------------
    # Extract/filter features
    # --------------------------------------------------------

    features = extract_features(
        contours,
        frame_width,
        frame_height
    )

    print(
        "Candidate contours after filtering:",
        len(features)
    )

    # --------------------------------------------------------
    # Print feature information
    # --------------------------------------------------------

    print("\nDetected candidate features:")
    print("-" * 100)

    if len(features) == 0:

        print("No candidate objects found.")

    else:

        for item in features:

            print(
                f"ID: {item['contour_id']:2d} | "
                f"Area: {item['area']:9.1f} | "
                f"Width: {item['width']:4d} | "
                f"Height: {item['height']:4d} | "
                f"Aspect: {item['aspect_ratio']:.2f} | "
                f"Center: "
                f"({item['center_x']:.1f}, "
                f"{item['center_y']:.1f})"
            )

    print("-" * 100)

    # --------------------------------------------------------
    # Draw candidate objects
    # --------------------------------------------------------

    result = draw_candidates(
        frame,
        contours,
        features
    )

    # --------------------------------------------------------
    # Display images
    # --------------------------------------------------------

    cv2.imshow(
        "Original Frame",
        frame
    )

    cv2.imshow(
        "Grayscale",
        gray
    )

    cv2.imshow(
        "Threshold",
        binary
    )

    cv2.imshow(
        "Candidate Fly Objects",
        result
    )

    print("\nWindows opened successfully.")
    print("Press Q to close.")

    # --------------------------------------------------------
    # Wait for Q
    # --------------------------------------------------------

    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    print("\nPreprocessing completed.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()