#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:50:28 2023

@author: Martin
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from collections import namedtuple
from uldaq import (get_daq_device_inventory,
                   DaqDevice,
                   AInScanFlag,
				   DaqEventType,
                   WaitType,
                   ScanOption,
                   InterfaceType,
				   AiInputMode,
                   create_float_buffer,
				   AOutScanFlag,
                   IepeMode,
                   CouplingMode,
                   Range,
                   TriggerType
                   )



class Daq:
    def __init__(self,
                  fSample = 1000,                         # Default: Sampling frequency of 1000 Hz
                  iepePerChannel = [0, 0, 0, 0],             # Default: IEPE/ICP deactivated
                  couplingPerChannel = [1, 1, 1, 1],         # Default: AC Coupling on each channel
                  sensorSensitivityPerChannel = [1, 1, 1, 1],# Default: 1 V/meas on each channel
                  triggerType = 2,
                  numberOfInputs = 4,                        # Default: All 4 inputs active
                  numberOfOutputs = 1,                       # Default: 1 output
                  inputMaxRange = 10                         # Default: Full scale is 10 V on each channel.
                  ):
        self.fSample = fSample
        self.iepeActive = iepePerChannel
        self.couplingType = couplingPerChannel
        self.sensorSensitivity = sensorSensitivityPerChannel
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.inputMaxRange = inputMaxRange
        self.scanParams = namedtuple('ScanParams', 'buffer high_chan low_chan')	# tuple nomé
        
        self.interfaceType = InterfaceType.USB
        self.scanOptionsInput = ScanOption.DEFAULTIO
        
        self.fSample = fSample

        # Parameters for output
        self.voltageRangeIndex = 0  # Use the first supported range
        
        # Parameters for input
        self.eventTypes = DaqEventType.ON_DATA_AVAILABLE | DaqEventType.ON_END_OF_INPUT_SCAN | DaqEventType.ON_INPUT_SCAN_ERROR
    
    def listDevices(self):
        try:
            self.device = get_daq_device_inventory(self.interfaceType)
            self.numberOfDevices = len(self.device)
            if self.numberOfDevices == 0:
                raise Exception('Error: No DAQ devices found')
            print("\nFound " + str(self.numberOfDevices) + " DAQ device(s): ")
            for i in range(self.numberOfDevices):
                print("	",self.device[i].product_name, " (", self.device[i].unique_id, ")", " Number board: ", i, sep="")
        except (KeyboardInterrupt, ValueError):
            print("Could not find DAQ device(s).")
            
            
    def connect(self,Boardnum = 0):
		# Connects to DAQ device
        try:
            self.daqDevice = DaqDevice(self.device[Boardnum])
			# Connect to DAQ device
            descriptor = self.daqDevice.get_descriptor()
            self.daqDevice.connect()
            print("\nConnected to ", descriptor.dev_string,"\n")
        except (KeyboardInterrupt, ValueError):
            print("Could not connect DAQ device.")
            
    def disconnect(self, Boardnum = 0):
		# Disconnects DAQ device
        if self.daqDevice:
            if self.daqDevice.is_connected():
                self.daqDevice.disconnect()
                print("\nDAQ device", self.device[Boardnum].product_name, "is disconnected.")
            print("\nDaQ device", self.device[Boardnum].product_name, "is released.")
            self.daqDevice.release()
        
            
    def selectIepe(self, iepeActive = 0):
        if(iepeActive==0):
            self.iepeMode = IepeMode.DISABLED
        elif (iepeActive==1):
            self.iepeMode = IepeMode.ENABLED
		
    def selectCoupling(self, couplingType = 1):
        if(couplingType==0):
            self.couplingMode = CouplingMode.DC
        elif (couplingType==1):
            self.couplingMode = CouplingMode.AC
            
    def selectInputRange(self, inputMaxRange = 10):
        if(inputMaxRange==1):
            self.inputRange = Range.BIP1VOLTS
        elif (inputMaxRange==10):
            self.inputRange = Range.BIP10VOLTS
            
            
    def AIconfig(self):
		# Reads input analog data from specified channel
		# Get the AiDevice object and verify that it is valid.
        self.aiDevice = self.daqDevice.get_ai_device()
        self.inputMode = AiInputMode.SINGLE_ENDED
        self.aiConfig = self.aiDevice.get_config()
        self.aiInfo = self.aiDevice.get_info()
        
        # Here we set the IEPE, coupling and Sensitivity for each channel used
        for chan in range(self.numberOfInputs):
            self.selectIepe(self.iepeActive[chan])
            self.aiConfig.set_chan_iepe_mode(chan,self.iepeMode)
            
            self.selectCoupling(self.couplingType[chan])
            self.aiConfig.set_chan_coupling_mode(chan, self.couplingMode)

            self.aiConfig.set_chan_sensor_sensitivity(chan,
                                                       self.sensorSensitivity[chan]
                                                       )
        self.supportedInputRanges = self.aiInfo.get_ranges(self.inputMode)
        
    def AOutConfig(self):
		# Verify the specified DAQ device supports analog output.
        self.aoDevice = self.daqDevice.get_ao_device()

		# Verify the specified DAQ device supports hardware pacing for analog output.
        self.aoInfo = self.aoDevice.get_info()

		# Select the voltage range
        self.voltageRange = self.aoInfo.get_ranges()[0]
    
        
    def AOutSignal(self, outputSignal):        
		# Create a buffer for output data.
        self.outBuffer = create_float_buffer(self.numberOfOutputs, outputSignal.size)
        self.outBuffer = outputSignal
        self.bufferLength = int(outputSignal.size)        
        self.outputBuffer = create_float_buffer(1, self.bufferLength)
        
        for bufferIter in range(0,self.bufferLength,1):
            self.outputBuffer[bufferIter] = outputSignal[bufferIter]
                    
        
    def AInOutScan(self):
        self.scanOptions = ScanOption.BLOCKIO 		# option extrigger active connector extrigger and level trigger channel 
        
        try:
            # Allocate a buffer to receive the data.
            self.data = create_float_buffer(self.numberOfInputs, self.bufferLength)
            #self.trigger(self.triggerType)
            highChannelAi = int(self.numberOfInputs-1)
            lowChannelAi = int(0)


			# Store the scan event parameters for use in the callback function.
            self.scanEventParameters = self.scanParams(self.data, highChannelAi, lowChannelAi)

			# Enable the event to be notified every time 100 samples are available.
            self.availableSampleCount = 100
            self.daqDevice.enable_event(self.eventTypes, self.availableSampleCount,
                                        self.eventCallbackFunction,
									    self.scanEventParameters)

			# Start the finite acquisition.
            self.aoDevice.a_out_scan(0,
                                     0,
                                     Range.BIP3VOLTS,
                                     self.bufferLength,
                                     self.fSample,
                                     self.scanOptions,
                                     AOutScanFlag.DEFAULT,
                                     self.outputBuffer
                                     )
            
            self.selectInputRange(self.inputMaxRange)

            
            self.aiDevice.a_in_scan(0,
                                    highChannelAi,
                                    self.inputMode,
                                    self.inputRange,
                                    self.bufferLength,
								    self.fSample,
                                    self.scanOptions,
                                    AInScanFlag.DEFAULT,
                                    self.data)

			# Wait here until the scan is done ... events will be handled in the event handler
			# (eventCallbackFunction) until the background scan completes
            time_to_wait = -1.0
            self.aiDevice.scan_wait(WaitType.WAIT_UNTIL_DONE, time_to_wait)

        except (ValueError, NameError, SyntaxError):
            pass
        except Exception as e:
            print('\n', e)
        finally:
            if self.daqDevice:
				# Stop the acquisition if it is still running.
                if self.aiDevice and self.aiInfo and self.aiInfo.has_pacer():
                    self.aiDevice.scan_stop()
                    self.aoDevice.scan_stop()
                self.daqDevice.disable_event(self.eventTypes)

    def eventCallbackFunction(self, event_callback_args):
        event_type = event_callback_args.event_type
        if event_type == DaqEventType.ON_END_OF_INPUT_SCAN:
            print('\nThe scan is complete\n')
            
            
#-----------------------------------------
class Signal:
    def __init__(self,
                 fSample = 1000, # Default: Sampling frequency of 1000 Hz
                 SigType = "sine",  # Default: Signal is a sine
                 A = 0.1, # Default: Amplitude is 0.1 V
                 fPrim = 20, # Default: principal frequency is 20 Hz
                 fSec = 20000, # Default: secondary frequency is 20000 Hz
                 DC = 0, # Default: no DC offset
                 fade = [100,100], # Default: 100 points for fade_in and fade_out
                 T_max = 2,         # Default: 2 s of signal
                 ZeroPad = 0    # Default: 0 points of zero padding for latency
                 ):
        self.A = A
        self.fSample = fSample
        self.fPrim = fPrim
        self.fSec = fSec
        self.ZeroPad = ZeroPad
        self.T_max = T_max
        self.fade = fade
        self.SigType = SigType
        
    def t_axis(self):
        """Creates time axes for the signal"""
        return np.arange(self.T_max*self.fSample + self.ZeroPad)/self.fSample - self.ZeroPad/self.fSample
    
    def f_domain(self,s,N_fft):
        f_axis = np.fft.rfftfreq(N_fft,d=1/self.fSample)
        f_content = np.fft.rfft(s,N_fft)/self.fSample
        return [f_axis, f_content]
    
    def SSS_analysis(self,y,N_fft):
        """Computes the FFT of signal and frequency axix via self.f_domain
        Divides by analytical equation of X to avoid division by 0"""
        f_axis, Y = self.f_domain(y,N_fft)
        L = self.T_max/np.log(self.fSec/self.fPrim)
        # definition of the inferse filter in spectral domain Xinv
        # (Novak et al., "Synchronized swept-sine: Theory, application, and implementation."
        # Journal of the Audio Engineering Society 63.10 (2015): 786-798.
        # Eq.(43))
        Xinv = 2*np.sqrt(f_axis/L)*np.exp(-1j*2*np.pi * f_axis*L*(1-np.log(f_axis/self.fPrim)) + 1j*np.pi/4)
        Xinv[0] = 0j
        
        return Y/Xinv
    
    def pad(self):
        return np.zeros(self.ZeroPad)
    
    def stimulus(self):
        t_sig = self.t_axis()[self.ZeroPad:]    # [:ZeroPad] is for the padding, [ZeroPad:] is the useful signal
        
        if self.SigType == "sine":
            OutputSig = self.A * np.sin(2*np.pi*self.fPrim*t_sig)
            
        if self.SigType == "step":
            OutputSig = self.A * np.ones(len(t_sig))
            
        if self.SigType == "SSS":
            L = self.T_max/np.log(self.fSec/self.fPrim)
            OutputSig = np.sin(2*np.pi*self.fPrim*L*np.exp(t_sig/L))
        
        
        if self.fade[0]>0:
            fade_in = (-np.cos(np.arange(self.fade[0])/self.fade[0]*np.pi)+1) / 2
            OutputSig[:int(self.fade[0])] *= fade_in
        if self.fade[1]>0:
            fade_out = (np.cos(np.arange(self.fade[1])/self.fade[1]*np.pi)+1) / 2
            OutputSig[-int(self.fade[1]):] *= fade_out
            
        OutputSig = np.concatenate((self.pad(),OutputSig))
        
        return OutputSig
        
        
#-----------------------------------------

# class Model:
#     def __init__(self,
#                  fSample = fSample, # Default: Sampling frequency of 1000 Hz
#                  ):
    
#%%
plt.close('all')
if __name__ == "__main__":
    fs = 48e3
    f1 = 10
    f2 = 2000
    T = 2
    #-------------------Signal Tests
    sig = Signal(fSample=fs,A=.1,T_max=T,fPrim=f1,fSec=f2,fade=[.1*fs,.1*fs],SigType="SSS")
    
    # fig, ax = plt.subplots()
    # ax.plot(sig.t_axis(),sig.stimulus())
    # fig.show()
    
    
    # # Add a spectrogram to see the signal
    # [t_test,f_test,Sxx] = ss.spectrogram(sig.stimulus(),48e3)
    # fig2,ax2 = plt.subplots()
    # ax2.pcolormesh(f_test,t_test,Sxx)    

    # FRF = sig.SSS_analysis(sig.stimulus(),len(sig.stimulus()))
    # f_k = sig.f_domain(sig.stimulus(), len(sig.stimulus()))[0]
    
    # fig3,ax3 = plt.subplots()
    # ax3.semilogx(f_k,20*np.log(FRF))
    
    
    plt.show()
    
    #-------------------DAQ tests
    DT = Daq(fSample = fs,                                      # Default: Sampling frequency of 1000 Hz
                  iepePerChannel = [0, 0, 0, 0],                # Default: IEPE/ICP deactivated
                  couplingPerChannel = [1, 1, 1, 1],            # Default: AC Coupling on each channel
                  sensorSensitivityPerChannel = [1.014, 1.004, 1, 1],   # Default: 1 V/meas on each channel
                  triggerType = 2,
                  numberOfInputs = 4,                           # Default: All 4 inputs active
                  numberOfOutputs = 1,                          # Default: 1 output
                  inputMaxRange = 10                            # Default: Full scale is 10 V on each channel.
                  )
    DT.listDevices()
    
    DT.connect(Boardnum=0)
    
    DT.AOutSignal(sig.stimulus())
    
    DT.AIconfig()
    DT.AOutConfig()
    DT.AInOutScan()
    
    print(np.shape(DT.data))
    # databis = 
    fig,ax = plt.subplots()
    # ax.plot(DT.data[0::4])
    ax.plot(np.reshape(DT.data,(int(T*fs),4)))
    
    DT.disconnect(Boardnum=0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    