# -Eddy-current-simulation-with-a-falling-magnet-FEniCS

This is a basic simulation using FEniCS of a magnet falling on a metallic plate.

The simulation is made with the inbuilt mesh tools in FEniCS so that you do not have to install any new packages.

However, the solution uses the NVIDIA GPU and therefore you would need a NVIDIA GPU and CUPY for Python. If you do not have this GPU on your computer, you can use the Google Colab notebook using its GPU (refer to their tutorials on how to use it, its very easy to setup).

Otherwise, you would need to know some basics of FEniCS to understand the code. Please go through that over here - https://launchpadlibrarian.net/83776282/fenics-book-2011-10-27-final.pdf

Note: I use the legacy FEniCS and not their latest FEniCSx due to my personal preference. However, there is not much difference according to me. The method is the same, the functions are slightly different. Their community recommends you to use FEniCSx.

If you are new to FEA, I would highly recommend "Introduction to the Finite Element Method" book by J.N. Reddy.

Read the PDF file name "Electromagnetics-FEniCS-Background-doc.pdf" for the math background.
