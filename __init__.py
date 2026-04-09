# __init__.py
def classFactory(iface):
    from .plugin_main import SemanticSegEvalPlugin
    return SemanticSegEvalPlugin(iface)
