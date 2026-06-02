#!/bin/sh

USER_ID="$(id -u)"
USER=$(logname)

INSTALL_DIR=$(dirname "$0")

# Copy FlowBase.desktop (base only, 8 motors)
cp $INSTALL_DIR/FlowBase.desktop ~/Desktop/
gio set ~/Desktop/FlowBase.desktop metadata::trusted true
chmod +x ~/Desktop/FlowBase.desktop

# Copy FlowBaseGamepad.desktop (base only, 8 motors, gamepad teleop)
cp $INSTALL_DIR/FlowBaseGamepad.desktop ~/Desktop/
gio set ~/Desktop/FlowBaseGamepad.desktop metadata::trusted true
chmod +x ~/Desktop/FlowBaseGamepad.desktop

# Copy LinearRailVehicle.desktop (with linear rail, 9 motors)
cp $INSTALL_DIR/LinearRailVehicle.desktop ~/Desktop/
gio set ~/Desktop/LinearRailVehicle.desktop metadata::trusted true
chmod +x ~/Desktop/LinearRailVehicle.desktop

# Copy LinearRailVehicleGamepad.desktop (with linear rail, 9 motors, gamepad teleop)
cp $INSTALL_DIR/LinearRailVehicleGamepad.desktop ~/Desktop/
gio set ~/Desktop/LinearRailVehicleGamepad.desktop metadata::trusted true
chmod +x ~/Desktop/LinearRailVehicleGamepad.desktop
