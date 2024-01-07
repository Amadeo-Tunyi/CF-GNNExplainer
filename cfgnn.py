
import yaml
import subprocess

def load_config(file_path):
    with open(file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def apply_config(config):
    # Implement logic to apply configurations to your environment
    print("Applying configurations:")
    print(config)

    # Install packages if specified in the YAML file
    if 'packages' in config:
        print("Installing packages:")
        for package in config['packages']:
            install_package(package)

def install_package(package):
    try:
        subprocess.run(["pip", "install", package], check=True)
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")




        
if __name__ == "__main__":
    yaml_file_path = "C:/Users/amade/Downloads/cf-gnnexplainer-main/cf-gnnexplainer-main/environment.yml"
    config_data = load_config(yaml_file_path)
    apply_config(config_data)
