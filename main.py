import os
import json
import subprocess

# Sett arbeidskatalogen til prosjektroten
project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, 'data')

# Les konfigurasjonsfilen
config_path = os.path.join(project_root, 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)

symbol = config['symbol']

# Funksjon for å kjøre et skript med ekstra argumenter
def run_script(script_path, args=[]):
    result = subprocess.run(['python', script_path] + args, check=True)
    if result.returncode != 0:
        raise Exception(f"Script {script_path} failed")

if __name__ == "__main__":
    # Opprett data-katalogen hvis den ikke eksisterer
    os.makedirs(data_dir, exist_ok=True)

    # Liste over skript som skal kjøres i riktig rekkefølge
    scripts_to_run = [
        os.path.join(project_root, 'sentiment', 'fetch_news.py'),
        os.path.join(project_root, 'sentiment', 'analyze_sentiment.py'),
        os.path.join(project_root, 'prediction', 'historical_data.py'),
        os.path.join(project_root, 'prediction', 'prepare_data.py'),
        os.path.join(project_root, 'prediction', 'calculate_technical_indicators.py'),
        os.path.join(project_root, 'prediction', 'combine_data.py'),
        os.path.join(project_root, 'prediction', 'train_model.py'),
        os.path.join(project_root, 'prediction', 'predict.py'),
        os.path.join(project_root, 'open_ai_LLMS', 'trading_decision.py')
    ]

    for script in scripts_to_run:
        print(f"Kjører {script} med symbol {symbol}...")
        run_script(script, [symbol])
        print(f"Ferdig med {script}")
