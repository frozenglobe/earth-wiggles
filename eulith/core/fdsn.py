# FDSN and ObsPy inventory tools

import requests, json
import numpy as np
import pandas as pd
from obspy.clients.fdsn import Client, RoutingClient, header
from obspy.core.inventory import Channel, Station, Inventory

cred = '/space/jhyl3/.eidatoken'

def fdsn_registry_dict(debug=False):
    """
    Returns the dataselect URLs for each FDSN node, as a dictionary with node names as keys and URLs as values

    :type debug: bool
    :param debug: If True, prints the names of nodes that do not have a dataselect service. Default is False.
    :return: dictionary of FDSN node names and their corresponding dataselect URLs
    """
    fdsn_request = requests.get("https://www.fdsn.org/ws/datacenters/1/query?format=text")
    fdsn_json = json.loads(fdsn_request.text)
    fdsn_registry = []
    fdsn_error = []
    for t in fdsn_json['datacenters']:
        try:
            stn_idx = [i for i,s in enumerate(t['repositories'][0]['services']) if 'fdsnws-dataselect-1' in s['name']][0]
            url = t['repositories'][0]['services'][stn_idx]['url'].split('/fdsnws/dataselect')[0]
            if t['name'] == 'EarthScope': name = 'IRIS'
            elif t['name'] == 'EPOSFR': name = 'RESIF'
            else: name = t['name'].upper()
            fdsn_registry.append([name, url])
        except IndexError:
            fdsn_error.append(t['name'])
            continue
    out = dict(fdsn_registry)
    if debug: return (out, fdsn_error)
    else: return out

def check_server(node:str):
    """
    Quick way to check if server connection works, or needs debugging.
    Usually because a link has been changed. 0 = fine, 1 = dict failed, 2 = obspy failed

    :type node: str
    :param node: FDSN node name
    :return: 0 if connection is fine, 1 if dictionary lookup failed, 2 if ObsPy client failed
    """
    node = node.upper()
    try:
        Client(fdsn_registry_dict().get(node)); return 0
    except:
        try: 
            Client(node); return 1
        except: return 2

def fdsn_connect(node:str, cred:str=None):
    """
    Connects to an FDSN server and returns a client object.

    :type node: str
    :param node: FDSN node name
    :type cred: str
    :param cred: Path to EIDA credentials file
    :return: FDSN client object
    """
    node = node.upper()
    try: 
        client = Client(fdsn_registry_dict().get(node), eida_token=cred)
    except: 
        try: client = Client(node, eida_token=cred)
        except: raise Exception(f"Connection to {node} failed. Check node name and credentials.")
    return client

def fdsn_exception_lookup():
    """
    Returns exception status codes and respective explanations

    :return: dictionary of exception status codes and explanations
    """
    exception_classes = [s for s in dir(header) if (s.startswith('FDSN')) & s.endswith('Exception')]
    status_lookup = {s: r.status_code for s in exception_classes for r in [getattr(header, s)] if r.status_code is not None}
    return status_lookup
        
def adriaarray_nw(get=False, cred:str=None):
    """
    Returns networks within AdriaArray. 

    :type get: bool
    :param get: True: retrieves the list from the EIDA routing client. 
                False (default): returns a hardcoded list of networks.
    :type cred: str
    :param cred: path to EIDA credentials file. Required if get=True.
    :return: list of AdriaArray networks
    """
    if get: 
        if not cred: raise ValueError("EIDA credentials file path must be provided if get=True.")
        client = RoutingClient('eida-routing', credentials=cred)
        adriaarray = client.get_stations(network="_ADARRAY")
        aa_list = adriaarray.get_contents()['networks']

    if not get:
        aa_list = ['1Y', '2Y', '4P', '7B', '9H', 'AC', 'BS', 'BW', 'C4', 'CH', 'CL', 'CR', 'FR', 'G', \
                   'GR', 'GU', 'HA', 'HC', 'HL', 'HP', 'HT', 'IV', 'IX', 'IY', 'KO', 'LE', 'MD', 'MK', \
                   'ML', 'MN', 'MT', 'NI', 'OT', 'OX', 'PL', 'RD', 'RF', 'RO', 'SI', 'SJ', 'SK', 'SL', \
                   'ST', 'TV', 'UT', 'VM', 'VR', 'XP', 'Y5', 'Y8', 'Z6']

    return aa_list

def ws_stnreq_builder(node:str='IRIS',
                      base_url:str=None,
                      networks:list=None,
                      station:list=None,
                      level:str='channel',
                      startbefore:str=None,
                      endafter:str=None,
                      format:str='text'):
    """
    Builds a URL for querying station information from an FDSN server.

    :type node: str
    :param node: FDSN node name
    :type base_url: str
    :param base_url: Base URL for the FDSN node. If None, it will be retrieved from the FDSN registry.
    :type networks: list
    :param networks: List of network codes
    :type station: list
    :param station: List of station codes
    :type level: str
    :param level: Level of information to retrieve, e.g. channel, response
    :type startbefore: str
    :param startbefore: usually the end time for the query, e.g. '2024-12-31T23:59:59'
    :type endafter: str
    :param endafter: usually the start time for the query, e.g. '2020-01-01'
    :type format: str
    :param format: Format of the response. One of 'text', 'xml'
    :return: URL for the FDSN station query
    """
    if not base_url: base_url = fdsn_registry_dict().get(node.upper())
    url = f"{base_url}/fdsnws/station/1/query?format={format}"
    if level: url += f"&level={level}"
    if networks: url += f"&net={','.join(networks)}"
    if station: url += f"&sta={','.join(station)}"
    if startbefore: url += f"&startbefore={startbefore}"
    if endafter: url += f"&endafter={endafter}"

    return url

### station- and inventory-related tools

def convert_n1e2z3(chan_code:str, reverse:bool=False):
    """
    Converts channel codes between NEZ and 123. If reverse is True, converts from 123 to NEZ.

    :type chan_code: str
    :param chan_code: channel code to be converted
    :type reverse: bool
    :param reverse: If False (default), converts from NEZ to 123. If True, converts from 123 to NEZ.
    :return: converted channel code
    """
    chan_dict = {'N': '1', 'E': '2', 'Z': '3'}

    if not reverse:
        if chan_code in ['N', 'E', 'Z']: return chan_dict.get(chan_code)
        else: return chan_code
    if reverse:
        if chan_code in ['1', '2', '3']: return {v: k for k, v in chan_dict.items()}.get(chan_code)
        else: return chan_code

def get_lcp_from_chn(channel:Channel):
    """
    Returns the lower corner period of the response for a given ObsPy Channel object.
    Calculates by smallest pole.

    :type channel: obspy.core.inventory.channel.Channel
    :param channel: ObsPy Channel object
    :return: lower corner period in seconds
    """
    respobj = channel.response
    pz = respobj.get_paz()
    p_lc = min([abs(p) for p in pz.poles])  # pole that corresponds to lower corner
    T_lc = (2*np.pi) / p_lc
    return T_lc

def get_lcp_from_xml(root:requests.models.Response, nw:str, sta:str, band:str):
    """
    Given an XML root element, network code, station code, and band, 
    this function extracts the poles from the response and calculates 
    the lower corner period (T_lc) based on the minimum absolute value of the poles.
    
    :type root: requests.models.Response
    :param root: The root element of the parsed XML response.
    :type nw: str
    :param nw: network code
    :type sta: str
    :param sta: station code
    :type band: str
    :param band: band code (e.g., 'B', 'H', etc.)
    :return: lower corner period in seconds
    """
    nw_element = root.findall(f"./*[@code='{nw}']")[0]
    st_element = nw_element.findall(f"./*[@code='{sta}']")[0]
    ch_element = st_element.findall(f"./*[@code='{band}HZ']")[0]
    poles = [complex(float(x[0].text), float(x[1].text))
            for x in ch_element.findall(".//{http://www.fdsn.org/xml/station/1}Pole")]
    p_lc = min([np.abs(pole) for pole in poles])
    T_lc = (2*np.pi) / p_lc
    return T_lc

def get_resp_from_row(row:pd.Series, node_path=None):
    """
    note this used to be called 'get_resp', needs changing

    
    """
    if not node_path: client = fdsn_connect(row.node)
    else: client = Client(node_path)
    
    try:
        inv = client.get_stations(network=row.network,
                                  station=row.station,
                                  startbefore=row.req_start,
                                  endafter=row.req_end,
                                  channel=f'{row.channel[0]}H?',
                                  level='response')
        return inv
    except:
        print('Response retrieval failed.'); return