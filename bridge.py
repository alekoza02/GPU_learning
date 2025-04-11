import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'

class Bridge:
    def __init__(self, plot_results=True):

        self.plot_results = plot_results

        print("\n-------------")
        self.get_GPU_attributes(short=1)
        print("-------------\n")
        

    def load_kernel(self, kernel_name, function_name):
        # Initialization
        questions = ('Kernel name: ', 'Function name: ')
        answers = (kernel_name, function_name)

        print("\n---------\n")
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
            keys = ['MaxThreadsPerBlock', 'WarpSize', 'MaxThreadsPerMultiProcessor', 'MaxBlocksPerMultiprocessor', 'MaxRegistersPerBlock', 'MaxSharedMemoryPerBlock', 'MaxSharedMemoryPerBlockOptin', 'MultiProcessorCount', 'ClockRate']
            ris = cp.cuda.Device(0).attributes
            for key in keys:
                print(f"{key:40} -> {ris[key]:10}")

        else:
            for key, value in cp.cuda.Device(0).attributes.items():
                print(f"{key:40} -> {value:10}")

        self.ClockRate = ris['ClockRate']



    def start_timer(self):
        self.start = time.perf_counter()
        print(f'Execution started at: {time.strftime("%X")}')
    
    
    def stop_timer(self, precision=3):
        print(f'Execution took: {GREEN}{time.perf_counter() - self.start:.{precision}}s{RESET}')
    

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
        self._plot2D_ = plt.imshow(array)
    

    def show_plots2D(self, name):
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.title(f'Showing results of: {name}')
        plt.colorbar(self._plot2D_)
        plt.show()


    def close_exercise(self):
        # When you're done with the arrays
        # cp.cuda.memory_pool.free_all()
        return None


    def launch_exercise1(self):

        self.load_kernel('lezione1', 'add_arrays')

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

        if self.plot_results:
            self.plot1D(ris_x, 'array x') 
            self.plot1D(ris_y, 'array y') 
            self.plot1D(ris_z, 'array z') 
            self.show_plots1D()
        self.close_exercise()


    def launch_exercise2(self):
        
        self.load_kernel('lezione2', 'morph')

        def get_gaussian(w, h, amplitude=1.0, sigma=1.0):
            
            x = cp.linspace(0, w - 1, w)
            y = cp.linspace(0, h - 1, h)

            xv, yv = cp.meshgrid(x, y)

            # Center of the Gaussian
            xc = (w - 1) / 2
            yc = (h - 1) / 2

            gaussian = amplitude * cp.exp( - ((xv - xc) ** 2 + (yv - yc) ** 2) / (2 * sigma ** 2), dtype=cp.float32)

            return gaussian


        print("[INFO] Partial (1/6)")
        W_img, W_ker = 1024 + 512, 256 + 128
        W_delta = W_img - W_ker
        
        print("[INFO] Partial (2/6)")
        img_p1 = get_gaussian(W_img, W_img, 6, 128)
        img_p2 = get_gaussian(W_img, W_img, 1.5, 512)
        img = img_p1 + img_p2
        kernel = get_gaussian(W_ker, W_ker, 4, 128)
        output = cp.empty((W_delta, W_delta), dtype=cp.float32)

        print("[INFO] Partial (3/6)")
        block_size = (32, 32)  # Block size (may need tuning)
        grid_size = ((W_delta + block_size[0] - 1) // block_size[0],
                    (W_delta + block_size[1] - 1) // block_size[1])
        

        print("[INFO] Partial (4/6)")
        self.start_timer()
        self.imported_kernel(grid_size, block_size, (img.ravel(), kernel.ravel(), output.ravel(), W_img, W_img, W_ker, W_ker))
        cp.cuda.Device(0).synchronize()
        self.stop_timer()
        

        print("[INFO] Partial (5/6)")
        img_CPU = img.get()  
        kernel_CPU = kernel.get()  
        output_CPU = output.get() 


        print("[INFO] Partial (6/6)")
        if self.plot_results:
            self.plot2D(img_CPU) 
            self.show_plots2D('Image')
            self.plot2D(kernel_CPU) 
            self.show_plots2D('Kernel')
            self.plot2D(output_CPU) 
            self.show_plots2D('Tip')
        self.close_exercise()


    def launch_exercise3(self):
        
        self.load_kernel('lezione3', 'memory')

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
        time_GPU = cp.zeros((W_delta, W_delta), dtype=cp.float32)
        debug_GPU = cp.zeros((W_delta, W_delta), dtype=cp.uint32)

        print("[INFO] Partial (3/6)")
        block_size = (25, 25)  # Block size (may need tuning)
        grid_size = ((W_delta + block_size[0] - 1) // block_size[0],
                    (W_delta + block_size[1] - 1) // block_size[1])
        
        
        print("[INFO] Partial (4/6)")        
        self.start_timer()
        self.imported_kernel(grid_size, block_size, (img.ravel(), kernel.ravel(), output.ravel(), 
                                                        time_GPU.ravel(), debug_GPU.ravel(), 
                                                        W_img, W_img, W_ker, W_ker))
        cp.cuda.Device(0).synchronize()
        self.stop_timer()


        print("[INFO] Partial (5/6)")
        img_CPU = img.get()  
        kernel_CPU = kernel.get()  
        output_CPU = output.get() 
        time_CPU = time_GPU.get() 
        debug_CPU = debug_GPU.get() 

        print("[INFO] Partial (6/6)")
        if self.plot_results:
            self.plot2D(img_CPU) 
            self.show_plots2D('Image')
            self.plot2D(kernel_CPU) 
            self.show_plots2D('Kernel')
            self.plot2D(output_CPU) 
            self.show_plots2D('Tip')
            self.plot2D(debug_CPU) 
            self.show_plots2D('DEBUG INDICES')
            self.plot2D(time_CPU / self.ClockRate) 
            self.show_plots2D('DEBUG TIMES [ms]')
        self.close_exercise()
        