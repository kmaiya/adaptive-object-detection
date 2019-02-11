import subprocess

infile = 'engineering-fountain-outside-nonstationary/engineering-fountain-outside-nonstationary.MTS'
outfile = infile[:-3] + 'mp4'

subprocess.run(['ffmpeg', '-i', infile, outfile])
