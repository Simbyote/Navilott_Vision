# Step 1: Drive forward 1.0 second
execute_drive(motor, 0.40, 0.40, 1.3)
execute_drive(motor, 0.0, 0.0, 0.1)  # brief stop

# Step 2: Left turn sequence
execute_drive(motor, 0.36, 0.63, 2.75)
execute_drive(motor, 0.0, 0.0, 0.1)                

# Step 2: Right turn sequence (90° turn)
execute_drive(motor, 0.45, 0.0, 1.62)
execute_drive(motor, 0.0, 0.0, 0.1) 