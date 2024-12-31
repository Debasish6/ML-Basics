import subprocess

def get_wifi_passwords():
    # Run the command to get the list of Wi-Fi profiles
    profiles_data = subprocess.check_output(['netsh', 'wlan', 'show', 'profiles']).decode('utf-8', errors="backslashreplace").split('\n')
    
    profiles = [i.split(":")[1][1:-1] for i in profiles_data if "Airtel_vine_5569" in i]
    
    for profile in profiles:
        # Run the command to get the password for each profile
        profile_info = subprocess.check_output(['netsh', 'wlan', 'show', 'profile', profile, 'key=clear']).decode('utf-8', errors="backslashreplace").split('\n')
        
        try:
            # Extract the password from the profile info
            password = [b.split(":")[1][1:-1] for b in profile_info if "Key Content" in b][0]
        except IndexError:
            password = None
        
        print(f"Profile: {profile}\nPassword: {password}\n")

if __name__ == "__main__":
    get_wifi_passwords()
