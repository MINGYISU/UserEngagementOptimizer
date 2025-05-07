from preprocess_data import *
from train import out_mapping, feature_mapping, device, model
import torch

def predict_batch(features_list):
    """
    Make predictions for multiple properties
    
    Args:
        features_list: List of dictionaries, each containing feature values
    
    Returns:
        List of predicted rent values
    """
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Apply preprocessing
    X = df.apply(get_mapping, axis=1)
    X = X.to_numpy(np.float32)
    
    # Convert to tensor
    X = torch.from_numpy(X).to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(X)
    
    return outputs

reversed_out_mapping = {v: k for k, v in out_mapping.items()}
input_features = [
    {'event_properties': "{'line-of-business': 'general', 'account-id': '10475', '[Amplitude] Session Replay ID': 'a798db5f-d66a-468c-8e51-4f1412853e4d/1737021244277'}", 
    'user_properties': "{'businessUnit': [], 'roles': ['underwriter'], 'trackingVersion': '3.1', 'initial_twclid': 'EMPTY', 'initial_gclid': 'EMPTY', 'hostname': 'None', 'initial_utm_source': 'EMPTY', 'initial_dclid': 'EMPTY', 'initial_wbraid': 'EMPTY', 'initial_rtd_cid': 'EMPTY', 'initial_utm_id': 'EMPTY', 'initial_gbraid': 'EMPTY', 'initial_msclkid': 'EMPTY', 'initial_ttclid': 'EMPTY', 'initial_ko_click_id': 'EMPTY', 'initial_utm_medium': 'EMPTY', 'initial_referring_domain': 'EMPTY', 'initial_utm_content': 'EMPTY', 'isInternalUser': 'False', 'initial_utm_campaign': 'EMPTY', 'initial_li_fat_id': 'EMPTY', 'referrer': 'https://accounts.google.co.in/', 'initial_fbclid': 'EMPTY', 'initial_referrer': 'EMPTY', 'initial_utm_term': 'EMPTY', 'referring_domain': 'accounts.google.co.in'}", 
    'device_family': 'Windows', 
    'device_type': 'Windows', 
    'language': 'English', 
    'os_name': 'Chrome', 
    'city': 'Los Angeles', 
    'country': 'United States', 
    'region': 'California'}
]

results = []
predicted_actions = predict_batch(input_features)
for input, output in zip(input_features, predicted_actions):
    input['predicted_action'] = reversed_out_mapping[output.item()]
    results.append(input)

# save
import json
with open('results.json', 'w') as f:
    json.dump(results, f)