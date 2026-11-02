"""Map the user-provided inventory to provisional service IDs by exact name."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Transcribed from the GPU audit output supplied by the user on 2026-10-10.
INVENTORY = '''기원문 61
건숙문 47
경안문 34
인수문 39
공묵재 52
숙문당 46
태원전 100
건길문 47
영사재 36
보강문 48
홍경문 62
일중문 66
필성문 46
장안당 74
초양문 50
건청궁 56
인유문 19
정시합 53
복수당 74
녹금당 110
곤녕합 50
협길당 77
집옥재 60
향원정 181
경회루 96
수정전 52
함화당 62
흥복전 98
함원전 60
흠경각 64
근정전 89
용성문 36
협생문 40
유화문 81
기별청 26
교태전 66
양의문 39
경성전 84
응지당 51
강녕전 82
만춘전 74
사정전 75
연길당 54
연생전 97
근정문 19
천추전 87
흥례문 65
자선당 69
비현각 61
중광문 32
광화문 39
집경당 58
제수합 26
계조당 17
원길헌 22
건춘문 21
자경전 44
팔우정 26
함홍각 19
건순각 19'''


def main():
    source = ROOT.parent / 'gunggeum_entity_linker/data/entities.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    buildings = [e for e in data['entities'] if e['kind'] == 'building']
    rows = []
    for number, line in enumerate(INVENTORY.splitlines(), 1):
        name, count = line.split()
        candidates = [e for e in buildings if e['name_ko'] == name]
        if len(candidates) != 1:
            raise ValueError('Non-unique mapping: ' + name)
        rows.append({'dataset_folder': f'{number:02d}_{name}', 'name_ko': name,
                     'image_count': int(count), 'provisional_service_id': candidates[0]['id'],
                     'mapping_method': 'exact_name', 'status': 'pending_team_id_agreement'})
    assert len(rows) == 60 and sum(r['image_count'] for r in rows) == 3438
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    with (out / 'building_mapping.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mapped = {r['provisional_service_id'] for r in rows}
    report = {'source': 'user pasted GPU audit, 2026-10-10',
              'service_sources': data['sources'], 'mapped_classes': len(rows),
              'image_count': sum(r['image_count'] for r in rows),
              'service_entries_without_dataset_class': [
                  {'id': e['id'], 'name': e['name_ko']} for e in buildings if e['id'] not in mapped],
              'note': 'Name-based provisional mapping, not visual verification or official ID approval.'}
    (out / 'mapping_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
