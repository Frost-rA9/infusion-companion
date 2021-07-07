import sys
import usb.core


# find USB devices
dev = usb.core.find(find_all=True)

# loop
for cfg in dev:
    print(cfg)