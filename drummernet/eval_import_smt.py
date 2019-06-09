import os
import sys
import xml.dom.minidom
import shutil
from globals import *


def get_onsets(xml_annotation_file):
    """
    Reads an xml annotation file (as provided by the SMT Drums dataset)
    """
    onsets = {
        'HH': np.array([], dtype=np.int32),
        'KD': np.array([], dtype=np.int32),
        'SD': np.array([], dtype=np.int32)
    }

    doc = xml.dom.minidom.parse(xml_annotation_file)
    events = doc.getElementsByTagName('event')

    for event in events:
        sec = event.getElementsByTagName('onsetSec')[0].firstChild.data  # extract the annotated time
        instrument = event.getElementsByTagName('instrument')[0].firstChild.data  # extract the instrument

        try:
            onsets[instrument] = np.append(onsets[instrument], float(sec))
        except KeyError as e:
            sys.stderr.write('Error reading xml file: unkown instrument '' + instrument + ''.\n')
            raise type(e)(e.message + ', unkown instrument '' + instrument + ''.')

    return onsets['HH'], onsets['KD'], onsets['SD']


if __name__ == '__main__':
    # convert xml -> txt file for SMT
    smt_unzip = sys.argv[1]

    print('Processing annotations...')
    smt_drums_folder = os.path.join('../data_evals', 'SMT_DRUMS')
    os.makedirs(os.path.join(smt_drums_folder, 'annotations'), exist_ok=True)

    xml_files = os.listdir(os.path.join(smt_unzip, 'annotation_xml'))
    xml_files = [f for f in xml_files if f.endswith('xml')]
    for fn in xml_files:
        path = os.path.join(smt_drums_folder, 'annotation_xml', fn)
        onsets = {
            'HH': np.array([], dtype=np.int32),
            'KD': np.array([], dtype=np.int32),
            'SD': np.array([], dtype=np.int32),
        }

        onsets['HH'], onsets['KD'], onsets['SD'] = get_onsets(os.path.join(smt_unzip, 'annotation_xml', fn))

        with open(os.path.join(smt_drums_folder, 'annotations', fn.replace('.xml', '.txt')), 'w') as f:
            for key in onsets:
                for t in onsets[key]:
                    f.write('%f\t%s\n' % (t, key))

    print('Processing audio files - copying them..')
    audio_path = os.path.join(smt_drums_folder, 'audio')
    if os.path.exists(audio_path):
        shutil.rmtree(audio_path)
    os.makedirs(audio_path, exist_ok=True)

    for filename in os.listdir(os.path.join(smt_unzip, 'audio')):
        if 'MIX' in filename:
            shutil.copy(os.path.join(smt_unzip, 'audio', filename),
                        os.path.join(audio_path, filename))
    print('all done! check out if everything is fine at %s' % smt_drums_folder)
