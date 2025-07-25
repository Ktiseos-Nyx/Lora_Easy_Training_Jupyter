# widgets/calculator_widget.py
import ipywidgets as widgets
from IPython.display import display, Image
import os

class CalculatorWidget:
    def __init__(self):
        self.create_widgets()

    def create_widgets(self):
        """Creates the UI components for the LoRA Step Calculator."""
        header_icon = "🧮"
        header_main = widgets.HTML(f"<h2>{header_icon} LoRA Step Calculator</h2>")

        # --- Input Section ---
        input_desc = widgets.HTML("<h3>▶️ Training Parameters</h3><p>Enter your dataset and training settings to calculate the total training steps.</p>")

        self.image_count = widgets.IntText(
            description="Image Count:",
            value=100,
            style={'description_width': 'initial'}
        )

        self.repeats = widgets.IntText(
            description="Repeats:",
            value=10,
            style={'description_width': 'initial'}
        )

        self.epochs = widgets.IntText(
            description="Epochs:",
            value=10,
            style={'description_width': 'initial'}
        )

        self.batch_size = widgets.IntText(
            description="Batch Size:",
            value=1,
            style={'description_width': 'initial'}
        )

        self.calculate_button = widgets.Button(description="Calculate Steps", button_style='success')
        
        input_box = widgets.VBox([
            input_desc,
            self.image_count,
            self.repeats,
            self.epochs,
            self.batch_size,
            self.calculate_button
        ])

        # --- Output Section ---
        self.output_area = widgets.Output(layout=widgets.Layout(height='auto', overflow='auto', border='1px solid #ddd'))

        # --- Main Widget Box ---
        self.widget_box = widgets.VBox([header_main, input_box, self.output_area])

        # --- Button Events ---
        self.calculate_button.on_click(self.run_calculation)

    def run_calculation(self, b):
        self.output_area.clear_output()

        with self.output_area:
            images = self.image_count.value
            repeats = self.repeats.value
            epochs = self.epochs.value
            batch_size = self.batch_size.value

            if batch_size <= 0:
                print("❌ Batch size must be greater than zero.")
                return

            total_steps = (images * repeats * epochs) // batch_size

            # --- Display Results ---
            print("--- Calculation Results ---")
            print(f"📸 Images:       {images}")
            print(f"🔄 Repeats:      {repeats}")
            print(f"📅 Epochs:       {epochs}")
            print(f"📦 Batch Size:   {batch_size}")
            print("=" * 27)
            print(f"⚡ Total Steps:  {total_steps}")
            print("\n🎯 Calculation complete! Your training parameters look good.")

            # --- Display Doro Image ---
            image_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'doro.png')
            if os.path.exists(image_path):
                display(Image(filename=image_path))
            else:
                print(f"\n(Doro image not found at '{image_path}')")

    def display(self):
        display(self.widget_box)