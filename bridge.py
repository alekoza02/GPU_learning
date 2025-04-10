import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

class Bridge:
    def __init__(self):

        print("\n-------------")
        self.get_GPU_attributes()
        print("-------------\n")

        # Initialization
        input_mode = input("Input mode [A / m]:"); print("\n")
        questions = ('Kernel name: ', 'Function name: ')
        answers = ('lezione2', 'morph')

        if input_mode == "m":
            kernel_name = input(questions[0])
            function_name = input(questions[1])
        else:
            kernel_name, function_name = answers
            for question, answer in zip(questions, answers):
                print(question, answer)
        print("\n---------\n")

        # Leggi il file contenente il kernel CUDA
        with open(f'{kernel_name}.cu', 'r') as f:
            kernel_code = f.read()

        # Compilare il kernel
        from cupy import RawKernel
        self.imported_kernel = RawKernel(kernel_code, function_name)
        
        
    def get_GPU_attributes(self, short=True):
        if short:
            keys = ['MaxThreadsPerBlock', 'WarpSize', 'MaxThreadsPerMultiProcessor', 'MaxBlocksPerMultiprocessor', 'MaxRegistersPerBlock', 'MaxSharedMemoryPerBlock', 'MaxSharedMemoryPerBlockOptin', 'MultiProcessorCount']
            ris = cp.cuda.Device(0).attributes
            for key in keys:
                print(f"{key:40} -> {ris[key]:10}")

        else:
            for key, value in cp.cuda.Device(0).attributes.items():
                print(f"{key:40} -> {value:10}")



    def start_timer(self):
        self.start = time.perf_counter()
        print(f'Execution started at: {time.strftime("%X")}')
    
    
    def stop_timer(self, precision=3):
        print(f'Execution took: {time.perf_counter() - self.start:.{precision}}s')
    

    def plot1D(self, array, name):
        # Visualizzare il risultato
        plt.plot(array, label=name)


    def show_plots1D(self):
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Showing results of: NVidia CUDA code')
        plt.legend()
        plt.show()
    
    
    def plot2D(self, array):
        # Visualizzare il risultato
        plt.imshow(array)
    

    def show_plots2D(self, name):
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.title(f'Showing results of: {name}')
        plt.show()


    def launch_exercise1(self):

        self.start_timer()

        # Creare gli array su GPU
        n = 100000
        x = cp.random.rand(n, dtype=cp.float32)
        y = cp.random.rand(n, dtype=cp.float32)
        z = cp.zeros_like(x)

        block_size = 32
        grid_size = (n + block_size - 1) // block_size

        # Eseguire il kernel
        self.imported_kernel((grid_size,), (block_size,), (x, y, z, n))
        cp.cuda.Device(0).synchronize()

        # Spostiamo il risultato sulla CPU per visualizzarlo
        ris_x = x.get()  
        ris_y = y.get()  
        ris_z = z.get() 

        self.stop_timer()

        self.plot1D(ris_x, 'array x') 
        self.plot1D(ris_y, 'array y') 
        self.plot1D(ris_z, 'array z') 
        self.show_plots1D()

    
    def launch_exercise2(self):
        
        self.start_timer()

        def get_gaussian(w, h, amplitude=1.0, sigma=1.0):
            
            x = cp.linspace(0, w - 1, w, dtype=cp.float32)
            y = cp.linspace(0, h - 1, h, dtype=cp.float32)

            xv, yv = cp.meshgrid(x, y)

            # Center of the Gaussian
            xc = (w - 1) / 2
            yc = (h - 1) / 2

            gaussian = amplitude * cp.exp( - ((xv - xc) ** 2 + (yv - yc) ** 2) / (2 * sigma ** 2), dtype=cp.float32)

            return gaussian


        print("[INFO] Partial (1/6)")
        W_img, W_ker = 4096, 1024
        W_delta = W_img - W_ker
        
        print("[INFO] Partial (2/6)")
        img_p1 = get_gaussian(W_img, W_img, 6, 128)
        img_p2 = get_gaussian(W_img, W_img, 1.5, 512)
        img = img_p1 + img_p2
        kernel = get_gaussian(W_ker, W_ker, 4, 128)
        output = cp.zeros((W_delta, W_delta), dtype=cp.float32)

        print("[INFO] Partial (3/6)")
        block_size = (32, 32)  # Block size (may need tuning)
        grid_size = ((W_delta + block_size[0] - 1) // block_size[0],
                    (W_delta + block_size[1] - 1) // block_size[1])
        
        print("[INFO] Partial (4/6)")
        self.imported_kernel(grid_size, block_size, (img.ravel(), kernel.ravel(), output.ravel(), W_img, W_img, W_ker, W_ker))
        
        cp.cuda.Device(0).synchronize()

        print("[INFO] Partial (5/6)")
        img_CPU = img.get()  
        kernel_CPU = kernel.get()  
        output_CPU = output.get() 

        self.stop_timer()

        print("[INFO] Partial (6/6)")
        self.plot2D(img_CPU) 
        self.show_plots2D('Image')
        self.plot2D(kernel_CPU) 
        self.show_plots2D('Kernel')
        self.plot2D(output_CPU) 
        self.show_plots2D('Tip')