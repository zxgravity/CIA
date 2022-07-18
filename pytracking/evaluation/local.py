from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.network_path = 'networks/'    # Where tracking networks are stored.
    settings.results_path = 'tracking_results/'    # Where to store tracking results

    settings.got10k_path = 'Path to your got10k dataset'
    settings.tnl2k_path = 'Path to your tnl2k dataset'
    settings.lasot_path = 'Path to your lasot dataset'
    settings.otb_path = 'Path to your otb100 dataset'
    settings.trackingnet_path = 'Path to your trackingnet dataset'
    settings.vot_path = 'Path to your vot dataset'

    return settings

