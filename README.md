# pyLemur

The python implementation of the LEMUR method to analyze multi-condition single-cell data.


# Run code and debug

The notebooks folder contains quarto (.qmd) documents which allow me to experiment with implementations.

To start the debugger, run the following either in VSCode or in a script and then run the Remote Attach debugging
configuration from the Run and Debug menu in vscode

```python
import debugpy
debugpy.listen(5678)
print("Waiting for debugger attach")
debugpy.wait_for_client()
```
