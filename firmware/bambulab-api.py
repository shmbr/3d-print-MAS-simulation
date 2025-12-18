import time
import bambulabs_api as bl

IP = '192.168.1.102'
SERIAL = '03919C452407561'
ACCESS_CODE = '13127926'

if __name__ == '__main__':
    print('Starting bambulabs_api example')
    print('Connecting to BambuLab 3D printer')
    print(f'IP: {IP}')
    print(f'Serial: {SERIAL}')
    print(f'Access Code: {ACCESS_CODE}')

    # Create a new instance of the API
    printer = bl.Printer(IP, ACCESS_CODE, SERIAL)

    # Connect to the BambuLab 3D printer
    printer.connect()

    time.sleep(2)

    # Get the printer status
    status = printer.get_state()
    print(f'Printer status: {status}')

    print(f'Printer total_layer_num: {printer.total_layer_num()}')
    print('')

    test = printer.ams_hub()

    print(f'Printer total_layer_num: {test}')
    print('')
    print(f'Printer get_time: {printer.get_time()}')

    dump = printer.mqtt_dump()

    ams_hub = printer.ams_hub()
    print("AMS:")
    print(vars(ams_hub[0]))

    # Disconnect from the Bambulabs 3D printer
    printer.disconnect()
