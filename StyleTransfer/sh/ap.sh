killall hostapd
killall dnsmasq
killall dhcpcd
killall wpa_supplicant
ifconfig wlan0 192.168.3.1 up
hostapd -dddd -B /etc/wlan/hostapd.conf
dnsmasq -i wlan0 -l /data/dnsmasq.leases --no-daemon --no-resolv --no-poll --dhcp-range=192.168.3.100,192.168.3.200,1h &

