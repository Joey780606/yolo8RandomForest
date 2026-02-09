import customtkinter as ctk
from ui.FeatureExtractionPage import FeatureExtractionPage
from ui.TrainingPage import TrainingPage
from ui.DynamicRecognitionPage import DynamicRecognitionPage


class App(ctk.CTk):
    """Main application window with left sidebar navigation and 3 content pages."""

    def __init__(self, yoloDetector, rfTrainer, cameraCapture):
        super().__init__()

        self.yoloDetector = yoloDetector
        self.rfTrainer = rfTrainer
        self.cameraCapture = cameraCapture

        self.title("YOLOv8 + Random Forest Face Recognition")
        self.geometry("1200x700")
        self.minsize(1000, 600)

        # Configure grid: sidebar (col 0) + main content (col 1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._createSidebar()
        self._createPages()
        self._showPage("featureExtraction")

    def _createSidebar(self):
        """Create the left navigation sidebar."""
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        # Title
        titleLabel = ctk.CTkLabel(
            self.sidebar, text="Face Recognition",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        titleLabel.grid(row=0, column=0, padx=20, pady=(20, 30))

        # Navigation buttons
        self.navButtons = {}

        self.navButtons["featureExtraction"] = ctk.CTkButton(
            self.sidebar, text="1. Feature Extraction",
            command=lambda: self._showPage("featureExtraction"),
            font=ctk.CTkFont(size=14),
            height=40,
        )
        self.navButtons["featureExtraction"].grid(row=1, column=0, padx=15, pady=5, sticky="ew")

        self.navButtons["training"] = ctk.CTkButton(
            self.sidebar, text="2. RF Training",
            command=lambda: self._showPage("training"),
            font=ctk.CTkFont(size=14),
            height=40,
        )
        self.navButtons["training"].grid(row=2, column=0, padx=15, pady=5, sticky="ew")

        self.navButtons["recognition"] = ctk.CTkButton(
            self.sidebar, text="3. Dynamic Recognition",
            command=lambda: self._showPage("recognition"),
            font=ctk.CTkFont(size=14),
            height=40,
        )
        self.navButtons["recognition"].grid(row=3, column=0, padx=15, pady=5, sticky="ew")

        # Appearance mode selector at bottom
        appearanceLabel = ctk.CTkLabel(self.sidebar, text="Appearance:")
        appearanceLabel.grid(row=6, column=0, padx=20, pady=(10, 0))

        self.appearanceMenu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["System", "Dark", "Light"],
            command=self._changeAppearance,
        )
        self.appearanceMenu.grid(row=7, column=0, padx=20, pady=(5, 20))

    def _createPages(self):
        """Create all content pages."""
        self.pages = {}

        self.pages["featureExtraction"] = FeatureExtractionPage(
            self, self.yoloDetector,
        )
        self.pages["training"] = TrainingPage(
            self, self.rfTrainer,
        )
        self.pages["recognition"] = DynamicRecognitionPage(
            self, self.yoloDetector, self.rfTrainer, self.cameraCapture,
        )

        # Place all pages in column 1 but hide them initially
        for page in self.pages.values():
            page.grid(row=0, column=1, sticky="nsew")
            page.grid_remove()

        self.currentPage = None

    def _showPage(self, pageName):
        """Switch to the specified page."""
        if self.currentPage == pageName:
            return

        # Hide current page
        if self.currentPage is not None:
            self.pages[self.currentPage].grid_remove()
            # Notify page it's being hidden
            page = self.pages[self.currentPage]
            if hasattr(page, "onHide"):
                page.onHide()

        # Show new page
        self.pages[pageName].grid()
        self.currentPage = pageName

        # Notify page it's being shown
        page = self.pages[pageName]
        if hasattr(page, "onShow"):
            page.onShow()

        # Update button styles
        for name, btn in self.navButtons.items():
            if name == pageName:
                btn.configure(fg_color=("gray25", "gray75"))
            else:
                btn.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])

    def _changeAppearance(self, mode):
        """Change the application appearance mode."""
        ctk.set_appearance_mode(mode)

    def onClosing(self):
        """Clean up resources before closing."""
        # Stop camera if running
        if self.cameraCapture.isOpened():
            self.cameraCapture.stop()

        # Notify recognition page to stop
        recognitionPage = self.pages.get("recognition")
        if recognitionPage and hasattr(recognitionPage, "stopRecognition"):
            recognitionPage.stopRecognition()

        self.destroy()
