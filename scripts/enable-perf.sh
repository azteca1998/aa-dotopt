#!/bin/bash


# This command must be executed from the host.
sudo su -c "echo 0 > /proc/sys/kernel/perf_event_paranoid"
