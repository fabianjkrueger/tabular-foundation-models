import kagglehub

# FIXME
# FIND OUT HOW TO SPECIFY PATH WHERE TO SAVE THE DATA TO
# THEN OBVIOUSLY ADAPT THE CODE AND SAVE THE DATA TO
# DATA/RAW

# Download latest version
path = kagglehub.dataset_download("dhoogla/bccc-cpacket-cloud-ddos-2024")

print("Path to dataset files:", path)