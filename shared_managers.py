# shared_managers.py
# This file provides shared manager instances for use across widgets
# Import this in notebooks to get consistent, shared state

from core.managers import SetupManager, ModelManager
from core.dataset_manager import DatasetManager
from core.training_manager import HybridTrainingManager
from core.utilities_manager import UtilitiesManager

# Create single instances of each manager
# These will be shared across all widgets
setup_manager = SetupManager()
model_manager = ModelManager()
dataset_manager = DatasetManager(model_manager)
training_manager = HybridTrainingManager()
utilities_manager = UtilitiesManager()

# Import all widgets
from widgets.setup_widget import SetupWidget
from widgets.dataset_widget import DatasetWidget
from widgets.training_widget import TrainingWidget
from widgets.utilities_widget import UtilitiesWidget
from widgets.calculator_widget import CalculatorWidget

def create_widgets():
    """
    Create all widgets with shared manager instances
    This ensures all widgets use the same state
    """
    return {
        'setup': SetupWidget(setup_manager, model_manager),
        'dataset': DatasetWidget(dataset_manager),
        'training': TrainingWidget(training_manager),
        'utilities': UtilitiesWidget(utilities_manager),
        'calculator': CalculatorWidget()  # No manager needed
    }

def create_widget(widget_name):
    """Create a single widget with proper dependency injection"""
    widgets_map = {
        'setup': lambda: SetupWidget(setup_manager, model_manager),
        'dataset': lambda: DatasetWidget(dataset_manager),
        'training': lambda: TrainingWidget(training_manager),
        'utilities': lambda: UtilitiesWidget(utilities_manager),
        'calculator': lambda: CalculatorWidget()
    }
    
    if widget_name not in widgets_map:
        raise ValueError(f"Unknown widget: {widget_name}. Available: {list(widgets_map.keys())}")
    
    return widgets_map[widget_name]()