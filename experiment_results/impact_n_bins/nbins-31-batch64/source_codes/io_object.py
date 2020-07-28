import os, time, sys, datetime

class IO_Object(object):
      path=None
      src_path=None
      args=None
      def __init__(self,args):  
          # input: arguments from program input
          if (len(args.comment.strip()))>0:
             comment='-'.join(args.comment.split(' '))
             self.path=os.path.join(str(datetime.datetime.now()).replace(':','-').replace(' ','-')[:-7]+'-'+comment)
          else:
             self.path=str(datetime.datetime.now()).replace(':','-').replace(' ','-')[:-7]   
          self.path+='-bin%d-batch%d'%(args.n_bins,args.batch_size)
          self.path=os.path.join('Retrieval_result',self.path)
          self.src_path=os.path.join(self.path,'source_codes')

          self.args=args
      
      def makeFolder(self):
          assert self.path is not None
          assert self.src_path is not None
          if os.path.exists('Retrieval_result')==False:
             os.mkdir('Retrieval_result')
          os.mkdir(self.path)
          os.mkdir(self.src_path)        

      def backup(self):
          assert self.src_path is not None
          import shutil,glob

          for file_name in glob.glob(os.path.join(sys.path[0],'*.py')):
              shutil.copy(file_name,self.src_path)
          if self.args is not None:
             with open(os.path.join(self.path,'opts.txt'),'w',encoding='utf-8') as dest:
                  dest.write(str(self.args)+'\n')

      def writeList(self,results,fileName):
          with open(os.path.join(self.path,fileName),'w',encoding='utf-8') as dest:
               for item in results:
                   dest.write(str(item)+'\n')
      def writeRun(self,run_dict):
          with open(os.path.join(self.path,'runs.tsv'),'w',encoding='utf-8') as dest:
            list_run_items=list(run_dict.items())
            list_run_items.sort(key=lambda x:x[0])
            for qid,list_pair in list_run_items:
                rank=0
                for pair in list_pair:
                    rank+=1
                    dest.write('%s\tQ0\t%s\t%d\t%f\t%s\n'%(qid,pair[0],rank,pair[1],'srm'))
               