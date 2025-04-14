from bridge import Bridge
    

if __name__ == '__main__':

    DEVELOPING = 1
    BENCHMARKING = not DEVELOPING

    if DEVELOPING:
        b = Bridge(plot_results=1, do_log=1)

        b.profile(b.launch_exercise5, (2880, 1800), 1)
        b.print_mean_time()
        
    elif BENCHMARKING:
        b = Bridge(plot_results=0, do_log=0)

        b.profile(b.launch_exercise5, (2880, 1800), 32)
        b.plot_mean_time()
    