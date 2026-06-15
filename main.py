import cv2
import os
from detect import Detect
from cropper import Cropper

def process_video(video_path, ad_path):
    # 1. Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # We need a temporary file path to save frames so Detect() can read them
    temp_frame_path = "test_images/temp_frame_processing.jpg"

    # 2. Load the detection models ONCE before the loop
    print("Loading detection models...")
    detector = Detect(temp_frame_path)  # path is just for init, we'll update per frame
    print("Models loaded. Starting video processing. Press 'q' to quit.")

    # 3. The Frame Loop
    while cap.isOpened():
        ret, frame = cap.read()
        
        # If the video ends or frame drops, break the loop
        if not ret or frame is None:
            print("Video ended.")
            break
        
        # 4. Save the current frame to disk temporarily
        cv2.imwrite(temp_frame_path, frame)

        # 5. Initialize the Cropper for this specific frame, reusing the detector
        crop = Cropper(temp_frame_path, ad_path, detector=detector)
        
        # 6. Run the detection logic
        w_ad, h_ad, h_base, w_base, mask_binary, if_found = crop.process_ad()

        # 7. If the billboard is found, warp it. Otherwise, show original frame.
        if if_found:
            final_frame, warped_ad = crop.warp_ad()
            cv2.imshow("DAR Pipeline", final_frame)
        else:
            cv2.imshow("DAR Pipeline", frame)

        # 8. Listen for the 'q' key to stop the video gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. Clean up windows and delete the temporary frame file
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)


if __name__ == "__main__":
    base_dir = "/home/skm/Downloads/DAR"
    video_file = os.path.join(base_dir, "test_images/clip1.mp4")
    ad_file = os.path.join(base_dir, "test_images/zomato.jpeg")
    
    process_video(video_file, ad_file)