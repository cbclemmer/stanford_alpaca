This repository is forked from the official stanford alpaca repository.  
  
In this directory there are four shell scripts that will allow you to easily fine tune a new llama model:  

### fine_tune.sh
This will take your fine tune data and run the training program on the 7B llama model and place the new model in a new "fine_tuned_model" directory

### convert.sh
This will take your fine tuned model checkpoints (created after fine_tune.sh) and change it into the huggingface model file structure

### clean.sh
This will remove the fine tuned model checkpoints from the fine tuned model, **be sure to run convert.sh first or this will delete all of your fine tune data.**

### test.sh
This will test your fine tuned model in the "fine_tuned_model" directory using test_instruction.txt for the instruction and test_input.txt for the input. This will take about 5 minutes.

## Step-by-step guide
First create a new json file with your fine tune data, it must be in the form of:
```
[
	{
		"instruction": "foo",
		"input": "bar",
		"output": "baz"
	}
]
```
**Note:** This is technically not valid json because the whole file is an array instead of an object but python seems to be able to read the file this way fine.
  
This follows the original alpaca instruction format, the instruction parameter is optional, but more context is helpful even if it is the same for every completion.
  
Once the file is created, name it `fine_tune_data.json` and place it in the base stanford_alpaca directory.
  
Next, run the `fine_tune.sh` script. It will take a while to load so be patient. The fine tune progress will be displayed in the terminal once it starts.
  
After the process finishes, you will have a new "fine_tuned_model" folder in the stanford_alpaca directory. This is the new model.
  
The model is formatted for you to continue fine tuning later with new data if you want (saved as a "checkpoint").  
**Warning:** I have not tested if the scripts pick back up with the checkpoint yet!  
  
Next, run the `convert.sh` script to convert the checkpoint to a format that the transformers (huggingface) library can use.  
  
Once the model is converted the data will still be in the same "fine_tuned_model" folder, but the checkpoint data will still be there so that the conversion can be run again if it failed.  
  
Run the `clean.sh` script to remove the checkpoint data.  
  
The model is now ready to be tested, create a `test_instruction.txt` file with a sample instruction and a `test_input.txt` file with a sample input.  
  
Run the `test.sh` script. This will take about 5 minutes, mostly due to loading the tokenizer (not sure why...).  
  
Once it completes, both the input and the result will be displayed in the terminal. Ensure that the model does what it needs to.  
  
If the model works correctly, then you can zip the folder with `zip fine_tuned_model foo.zip` replacing "foo.zip" with whatever you want to name the model.  
  
The file will be about 25GB so transferring it off of the server will take a while.  
  
Transfer the file via sftp or move to your own folder on the DGX and unzip it where you need it.  
  
Refer to the transformers library documentation for directions on how to use a local model to do inference (running the model normally), or you can just copy the `test.sh` file and modify it for your needs, just be sure to only load the model once so the script does not take forever.  
