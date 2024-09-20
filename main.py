from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

def detect_vehicles(frame):
    results = model(frame)
    vehicle_count = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                vehicle_count += 1
    return vehicle_count

def calculate_density(vehicle_count):
    if vehicle_count < 5:
        return "Low"
    elif vehicle_count < 15:
        return "Medium"
    else:
        return "High"

def generate_signal(densities):
    green_time = {"Low": 30, "Medium": 45, "High": 60}
    return [green_time[density] for density in densities]

def main():
    cap = cv2.VideoCapture(0)  # Use system camera
    
    while True:
        densities = []
        for lane in range(4):
            print(f"Analyzing Lane {lane + 1}")
            # In a real system, you would physically or electronically adjust the camera here
            # For simulation, we'll just capture a new frame for each "lane"
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from camera")
                continue
            
            vehicle_count = detect_vehicles(frame)
            density = calculate_density(vehicle_count)
            densities.append(density)
            print(f"Lane {lane + 1}: {vehicle_count} vehicles, Density: {density}")
            
            # Display the frame with lane number
            cv2.putText(frame, f"Lane {lane + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Traffic Analysis', frame)
            cv2.waitKey(2000)  # Wait for 2 seconds before "switching" to next lane
        
        signals = generate_signal(densities)
        print("Generated signals (seconds of green time):", signals)
        
        # Simulate traffic light cycle
        for i, signal_time in enumerate(signals):
            print(f"Lane {i+1} Green for {signal_time} seconds")
            time.sleep(1)  # Simulating time passage (remove in production)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()