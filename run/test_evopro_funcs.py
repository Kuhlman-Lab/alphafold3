from evopro_utils import init_af3
import json

print('running init')
max_length = 150
f = init_af3(0, 'temp.txt', [max_length])
print('finished init')

json_dict = {
    "name": "HIS_noodle",
    "sequences": [{
        "protein": {
            "id": ["A"],
            "sequence": "H" * max_length
        }
    }],
    "modelSeeds": ['42']
}
s = json.dumps(json_dict)
print('running function returned by init')
result_dict = f(s)
print('got results', result_dict[0].keys())