import keyboard

print("Press SPACE...")

while True:
    if keyboard.is_pressed("space"):
        print("SPACE DETECTED")