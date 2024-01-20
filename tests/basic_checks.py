import unittest
import os
import importlib.util

class TestLLama2SFT(unittest.TestCase):

    def test_library_torch_installed(self):
        """ Test if pytorch library is installed """
        torch_installed = importlib.util.find_spec("torch") is not None
        self.assertTrue(torch_installed, "torch is not installed")

    def test_library_peft_installed(self):
        """ Test if peft library is installed """
        peft_installed = importlib.util.find_spec("peft") is not None
        self.assertTrue(peft_installed, "peft library is not installed")
        
    def test_library_datasets_installed(self):
        """ Test if datasets library is installed """
        datasets_installed = importlib.util.find_spec("datasets") is not None
        self.assertTrue(datasets_installed, "datasets library is not installed")
                
    def test_library_transformers_installed(self):
        """ Test if transformers library is installed """
        transformers_installed = importlib.util.find_spec("transformers") is not None
        self.assertTrue(transformers_installed, "transformers library is not installed")
        
    def test_library_bitsandbytes_installed(self):
        """ Test if bitsandbytes library is installed """
        bitsandbytes_installed = importlib.util.find_spec("bitsandbytes") is not None
        self.assertTrue(bitsandbytes_installed, "bitsandbytes is not installed")
                
    def test_library_ctranslate2_installed(self):
        """ Test if ctranslate2 library is installed """
        ctranslate2_installed = importlib.util.find_spec("ctranslate2") is not None
        self.assertTrue(ctranslate2_installed, "ctranslate2 is not installed")
                
    def test_library_streamlit_installed(self):
        """ Test if streamlit library is installed """
        streamlit_installed = importlib.util.find_spec("streamlit") is not None
        self.assertTrue(streamlit_installed, "streamlit is not installed")
                
    def test_library_mlflow_installed(self):
        """ Test if mlflow library is installed """
        mlflow_installed = importlib.util.find_spec("mlflow") is not None
        self.assertTrue(mlflow_installed, "mlflow is not installed")
                
    def test_library_trl_installed(self):
        """ Test if trl library is installed """
        trl_installed = importlib.util.find_spec("trl") is not None
        self.assertTrue(trl_installed, "trl is not installed")

if __name__ == '__main__':
    unittest.main()
