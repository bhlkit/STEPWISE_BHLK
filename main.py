# main.py
from gui import SplashScreen, StepwiseApp

if __name__ == "__main__":
    # 1. Jalankan Splash Screen dulu
    splash = SplashScreen()
    splash.mainloop()
    
    # 2. Setelah Splash selesai, baru jalankan App Utama
    app = StepwiseApp()
    app.mainloop()