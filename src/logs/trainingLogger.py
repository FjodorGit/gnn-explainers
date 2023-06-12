from pathlib import Path


class TrainingLogger():

    def __init__(self, logPath: str):
        self.logPath = logPath

    def log(self, message: str):
        print(message)
        with open(self.logPath, "+a") as logFile:
            logFile.write(message + "\n")
        return 
