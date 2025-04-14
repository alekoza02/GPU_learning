import cupy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'

class Bridge:
    def __init__(self, plot_results=True, do_log=True):

        self.plot_results = plot_results
        self.times = []
        self.do_log = do_log

        print("\n-------------")
        self.get_GPU_attributes(short=1)
        print("-------------\n")
        

    def load_kernel(self, kernel_name, function_name):
        # Initialization
        questions = ('Kernel name: ', 'Function name: ')
        answers = (kernel_name, function_name)

        self.log("\n---------\n")
        for question, answer in zip(questions, answers):
            self.log((question, answer))
        self.log("\n---------\n")

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
        self.log(f'Execution started at: {time.strftime("%X")}')
    
    
    def stop_timer(self, precision=3, save=False):
        exec_time = time.perf_counter() - self.start
        self.log(f'Execution took: {GREEN}{exec_time:.{precision}}s{RESET}')
        if save:
            self.times.append(exec_time)


    def log(self, text):
        if self.do_log:
            print(text)


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
    

    def show_plots2D(self, name, ZBAR=True):
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.title(f'Showing results of: {name}')
        if ZBAR: plt.colorbar(self._plot2D_)
        plt.show()


    def plot_mean_time(self, print_stats=True):
        self.analyze_data(print_stats)
        self.plot1D(self.times, "Execution times [s]")    
        media = sum(self.times) / len(self.times)
        self.plot1D([media for _ in self.times], f'Media: {media:.4f}')
        if print_stats: self.print_mean_time(False)
        self.show_plots1D()
        self.times = []


    def print_mean_time(self, print_stats=True):
        self.analyze_data(print_stats)
        media = sum(self.times) / len(self.times)
        print(f"\nMEAN: {media:.4f} | MAX: {max(self.times):.4f} | MIN: {min(self.times):.4f} | FPS: {1 / media:.2f}\n")
        self.times = []


    def analyze_data(self, print_stats):
        def mean_std(data):
            n = len(data)
            if n == 0:
                raise ValueError("Empty list")
            if n == 1:
                return 1, 1

            mean = sum(data) / n
            # Sample standard deviation
            variance = sum((x - mean) ** 2 for x in data) / (n - 1)
            std_dev = math.sqrt(variance)
            
            return mean, std_dev
        
        print_info1 = len(self.times)

        run = 1
        while run:
            m, s = mean_std(self.times)
            new_array = []
            old_len = len(self.times)
            for time in self.times:
                if (time - m) / s < 3: new_array.append(time) 
            self.times = new_array
            new_len = len(self.times)
            run = old_len != new_len

        if print_stats and print_info1 - len(self.times) > 0: print(f"Removed {print_info1 - len(self.times)} elements (OUTLIERS).")


    def profile(self, custom_foo, custom_args, num=10):
        def progress(value, max_value):
            percent = (value + 1) / max_value
            bar = ('#' * int(percent * 40)).ljust(40)
            sys.stdout.write(f'\r[{bar}] {percent*100:.1f}% ({value + 1} / {max_value})')
            sys.stdout.flush()
            if percent == 1:
                print(f"{RESET}")

        for i in range(num):
            custom_foo(*custom_args)
            progress(i, num)


    def close_exercise(self):
        # When you're done with the arrays
        # cp.cuda.memory_pool.free_all()
        return None


    def launch_exercise1(self, n=100000):

        self.load_kernel('lezione1', 'add_arrays')

        self.start_timer()

        # Creare gli array su GPU
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


    def launch_exercise2(self, img_size=512, ker_size=64):
        
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


        self.log("[INFO] Partial (1/6)")
        W_img, W_ker = img_size, ker_size
        W_delta = W_img - W_ker
        
        self.log("[INFO] Partial (2/6)")
        img_p1 = get_gaussian(W_img, W_img, 6, 128)
        img_p2 = get_gaussian(W_img, W_img, 1.5, 512)
        img = img_p1 + img_p2
        kernel = get_gaussian(W_ker, W_ker, 4, 128)
        output = cp.empty((W_delta, W_delta), dtype=cp.float32)

        self.log("[INFO] Partial (3/6)")
        block_size = (32, 32)  # Block size (may need tuning)
        grid_size = ((W_delta + block_size[0] - 1) // block_size[0],
                    (W_delta + block_size[1] - 1) // block_size[1])
        

        self.log("[INFO] Partial (4/6)")
        self.start_timer()
        self.imported_kernel(grid_size, block_size, (img.ravel(), kernel.ravel(), output.ravel(), W_img, W_img, W_ker, W_ker))
        cp.cuda.Device(0).synchronize()
        self.stop_timer(save=True)
        

        self.log("[INFO] Partial (5/6)")
        img_CPU = img.get()  
        kernel_CPU = kernel.get()  
        output_CPU = output.get() 


        self.log("[INFO] Partial (6/6)")
        if self.plot_results:
            self.plot2D(img_CPU) 
            self.show_plots2D('Image')
            self.plot2D(kernel_CPU) 
            self.show_plots2D('Kernel')
            self.plot2D(output_CPU) 
            self.show_plots2D('Tip')
        self.close_exercise()


    def launch_exercise3(self, img_size=512, ker_size=64):
        
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


        self.log("[INFO] Partial (1/6)")
        W_img, W_ker = img_size, ker_size
        W_delta = W_img - W_ker
        
        self.log("[INFO] Partial (2/6)")
        img_p1 = get_gaussian(W_img, W_img, 6, 128)
        img_p2 = get_gaussian(W_img, W_img, 1.5, 512)
        img = img_p1 + img_p2
        kernel = get_gaussian(W_ker, W_ker, 4, 128)
        output = cp.zeros((W_delta, W_delta), dtype=cp.float32)
        time_GPU = cp.zeros((W_delta, W_delta), dtype=cp.float32)
        debug_GPU = cp.zeros((W_delta, W_delta), dtype=cp.uint32)

        self.log("[INFO] Partial (3/6)")
        block_size = (32, 8)  # Block size (may need tuning)
        grid_size = ((W_delta + block_size[0] - 1) // block_size[0],
                    (W_delta + block_size[1] - 1) // block_size[1])
        
        
        self.log("[INFO] Partial (4/6)")        
        self.start_timer()
        self.imported_kernel(grid_size, block_size, (img.ravel(), kernel.ravel(), output.ravel(), 
                                                        time_GPU.ravel(), debug_GPU.ravel(), 
                                                        W_img, W_img, W_ker, W_ker))
        cp.cuda.Device(0).synchronize()
        self.stop_timer(save=True)


        self.log("[INFO] Partial (5/6)")
        img_CPU = img.get()  
        kernel_CPU = kernel.get()  
        output_CPU = output.get() 
        time_CPU = time_GPU.get() 
        debug_CPU = debug_GPU.get() 

        self.log("[INFO] Partial (6/6)")
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
        


    def launch_exercise4(self, W=2880, H=1800):
        
        self.load_kernel('lezione4', 'shader')

        
        self.log("[INFO] Partial (1/6)")
        W_img, H_img = W, H
        
        self.log("[INFO] Partial (2/6)")
        output = cp.empty((W_img, H_img, 3), dtype=cp.float32)
        
        self.log("[INFO] Partial (3/6)")
        block_size = (32, 32)  # Block size (may need tuning)
        grid_size = ((W_img + block_size[0] - 1) // block_size[0],
                    (H_img + block_size[1] - 1) // block_size[1])
        
        
        self.log("[INFO] Partial (4/6)")        
        self.start_timer()
        self.imported_kernel(grid_size, block_size, (output.ravel(), W_img, H_img))
        cp.cuda.Device(0).synchronize()
        self.stop_timer(save=True)


        self.log("[INFO] Partial (5/6)")
        output_CPU = output.get() 
        
        self.log("[INFO] Partial (6/6)")
        if self.plot_results:
            self.plot2D(output_CPU.transpose(1, 0, 2)) 
            self.show_plots2D('Shader', ZBAR=False)
        self.close_exercise()
        

    def launch_exercise5(self, W=2880, H=1800):
        
        self.load_kernel('lezione5', 'shader')

        self.log("[INFO] Partial (1/6)")
        W_img, H_img = W, H
        time = 0.1
        mouse_x, mouse_y = 300, 200

        self.log("[INFO] Partial (2/6)")
        output = cp.empty((W_img, H_img, 3), dtype=cp.float32)
        
        self.log("[INFO] Partial (3/6)")
        block_size = (32, 32)  # Block size (may need tuning)
        grid_size = ((W_img + block_size[0] - 1) // block_size[0],
                    (H_img + block_size[1] - 1) // block_size[1])
        
        
        self.log("[INFO] Partial (4/6)")        
        self.start_timer()
        self.imported_kernel(grid_size, block_size, (output.ravel(), cp.int32(W_img), cp.int32(H_img), cp.float32(time), cp.int32(mouse_x), cp.int32(mouse_y)))
        cp.cuda.Device(0).synchronize()
        self.stop_timer(save=True)


        self.log("[INFO] Partial (5/6)")
        output_CPU = output.get() 
        
        self.log("[INFO] Partial (6/6)")
        if self.plot_results:
            output_CPU = output_CPU.reshape((H, W, 3))
            self.plot2D(output_CPU) 
            self.show_plots2D('Shader', ZBAR=False)
        self.close_exercise()