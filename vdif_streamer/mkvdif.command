python -m cProfile -o profile_fake.out ./vdif_streamer.py /mnt/disk2/MeerKAT/vlbi_J0530+1331_array_scan08_bf_pol0_1519310448.h5 /mnt/disk2/MeerKAT/vlbi_J0530+1331_array_scan08_bf_pol1_1519310448.h5 -s 2018y053d14h48m23s -e 2018y053d14h48m28s /mnt/disk2/MeerKAT/n18l1_me_no0008_fake1.vdif -f 1658.49 -r 64
~/anaconda3/bin/python ~/pfbfft/vdif_streamer/vdif_streamer.py 1541081970_i0_tied_array_channelised_voltage_0x.h5 1541081970_i0_tied_array_channelised_voltage_0y.h5 -s 2018y305d14h20m0s -e  2018y305d14h20m10s /data/verkouter/n18l3_me_no0006.vdif -f 1658.49 -r 64
# Note: the no0006 was the scan from N18L1 schedule, in N18L3 the 14h20m0s-14h20m10s range is in scan No0018
#       so: $> mv /data/verkouter/n18l3_me_no0006.vdif /data/verkouter/n18l3_me_no0018.vdif
