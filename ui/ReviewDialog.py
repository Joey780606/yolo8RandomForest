import customtkinter as ctk


class ReviewDialog(ctk.CTkToplevel):
    """
    Modal dialog for reviewing YOLO face detection results.
    Offers 4 choices: Correct, Recapture, Discard this photo, System capture for all.
    """

    def __init__(self, parent, imageName=""):
        super().__init__(parent)

        self.result = None

        self.title("Review Detection")
        self.geometry("400x250")
        self.resizable(False, False)

        # Center the dialog on parent
        self.transient(parent)

        # Title label
        titleLabel = ctk.CTkLabel(
            self, text=f"Review: {imageName}",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        titleLabel.pack(padx=20, pady=(20, 5))

        infoLabel = ctk.CTkLabel(
            self, text="Are the detected landmarks correct?",
            font=ctk.CTkFont(size=13),
        )
        infoLabel.pack(padx=20, pady=(0, 15))

        # Button frame
        buttonFrame = ctk.CTkFrame(self, fg_color="transparent")
        buttonFrame.pack(padx=20, pady=10, fill="x")

        correctBtn = ctk.CTkButton(
            buttonFrame, text="Correct",
            command=lambda: self._setResult("correct"),
            fg_color="green", hover_color="darkgreen",
            height=35,
        )
        correctBtn.pack(fill="x", pady=3)

        recaptureBtn = ctk.CTkButton(
            buttonFrame, text="Recapture",
            command=lambda: self._setResult("recapture"),
            height=35,
        )
        recaptureBtn.pack(fill="x", pady=3)

        discardBtn = ctk.CTkButton(
            buttonFrame, text="Discard This Photo",
            command=lambda: self._setResult("discard"),
            fg_color="gray", hover_color="darkgray",
            height=35,
        )
        discardBtn.pack(fill="x", pady=3)

        systemCaptureBtn = ctk.CTkButton(
            buttonFrame, text="System Capture For All",
            command=lambda: self._setResult("systemCaptureAll"),
            fg_color="orange", hover_color="darkorange",
            height=35,
        )
        systemCaptureBtn.pack(fill="x", pady=3)

        # Make modal
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: self._setResult("discard"))

    def _setResult(self, result):
        """Set the dialog result and close."""
        self.result = result
        self.grab_release()
        self.destroy()

    def getResult(self):
        """Wait for the dialog to close and return the result."""
        self.wait_window()
        return self.result
