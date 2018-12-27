# !/usr/bin/sh

function update_jd_cv{
  hive -e "
  add jar /data1/nlp/oceanus-online/udf/lib/oceanus-common-1.0.8-SNAPSHOT.jar;
  add jar /data1/nlp/oceanus-online/udf/lib/oceanus-etl-hive-udf-1.0-SNAPSHOT.jar;
  add jar /data1/nlp/oceanus-online/udf/lib/oceanus-nltk-common-1.0.8-SNAPSHOT.jar;
  add jar /data1/nlp/oceanus-online/udf/lib/oceanus-nltk-segment-1.0.8-SNAPSHOT.jar;
  create temporary function tokenizer as 'com.techwolf.oceanus.etl.hive.udf.TokenizerArrayUDF';

  select '======================= selecting geek profile =============================' from dw_bosszp.zp_geek limit 1;

  create temporary table nlp.qyh_geek as
    select
      geek_id user_id, concat_ws('\t',collect_set(string(desc))) desc
    from
      dw_bosszp.zp_geek
    group by
      geek_id;

  create temporary table nlp.qyh_project as
    select
      user_id, concat_ws('\t',collect_set(string(description))) prj
    from
      dw_bosszp.zp_project
    group by
      user_id;

  create temporary table nlp.qyh_edu as
    select
      user_id, concat_ws('\t',collect_set(string(edu_desc))) edu
    from
      dw_bosszp.zp_edu
    group by
      user_id;

  create temporary table nlp.qyh_work as
    select
      user_id, concat_ws('\t', collect_set(memo)) workdesc
    from
      dw_bosszp.zp_work
    group by
      user_id;

  create temporary table nlp.qyh_resume as
    select
      a.user_id geek_id, a.desc, b.prj, c.edu, d.workdesc
    from
      nlp.qyh_geek a
    left outer join
      nlp.qyh_project b
    on
      a.user_id = b.user_id
    left outer join
      nlp.qyh_edu c
    on
      a.user_id = c.user_id
    left outer join
      nlp.qyh_work d
    on
      a.user_id = d.user_id;

  insert overwrite table nlp.geek_cv_token
    select 
      geek_id,
      concat_ws('\t', 
        tokenizer(
          split(
            regexp_replace(
              concat_ws('\t', desc, prj, edu, workdesc), 
              '([，、：“”‘（）《》〈〉【】『』「」﹃﹄〔〕…—～﹏￥]+)',
              ' '
            ),
            '([；。？！\t\n]+)|(([0-9]|[一二三四五六七八九])[.、 ])'
          )
        )
      ) cv
    from
      nlp.qyh_resume;

  select '======================= selecting job profile =============================' from dw_bosszp.zp_geek limit 1;

  insert overwrite table nlp.job_desc_token
    select
      id job_id,
      concat_ws('\t', 
        tokenizer(
          split(
            regexp_replace(
              concat_ws(
                '\t',
                collect_set(job_desc)
              ),
              '([，、；：“”‘（）《》〈〉【】『』「」﹃﹄〔〕…—～﹏￥]+)',
              ' '
            ),
            '([；。？！\t\n]+)|(([0-9]|[一二三四五六七八九])[.、 ])'
          )
        )
      ) jd
    from
      dw_bosszp.zp_job
    group by
      id;  
  select '======================= done =============================' from dw_bosszp.zp_geek limit 1;
  "
}

function select_add_no_chat(){
  fp=$1
  st=$2
  st=-5
  et=$[$st+1]
  etp=$[$et+3]
  
  start_time=$(date -d "$st day" +%Y-%m-%d)
  end_time=$(date -d "$et day" +%Y-%m-%d)
  end_time_plus=$(date -d "$etp day" +%Y-%m-%d)
  n_positive=$[$3/2]
  n_negative=$[$3/2]

  hive -e "
  select '======================= selecting all samples =============================' from dw_bosszp.zp_geek limit 1;

  create temporary table nlp.qyh_sample_geek_add as
    select 
      uid geek_id, actionp2 job_id, actionp boss_id, action
    from
      dw_bosszp.bg_action
    where
      ds > '$start_time' and ds <= '$end_time'
    and
      action = 'detail-geek-addfriend'
    and
      bg = 0
    group by
      uid, actionp2, actionp, action;

  select '======================= selecting chat samples =============================' from dw_bosszp.zp_geek limit 1;

  create temporary table nlp.qyh_sample_boss_chat as
    select 
      actionp geek_id, uid boss_id, action
    from
      dw_bosszp.bg_action
    where
      ds > '$start_time' and ds <= '$end_time_plus'
    and
      action = 'chat'
    and
      bg = 1
    group by 
      actionp, uid, action;

  select '======================= joining chat with add friend =============================' from dw_bosszp.zp_geek limit 1;

  create temporary table nlp.qyh_sample_geek_add_boss_chat as
    select 
      geek_id, job_id, action 
    from (
      select
        a.geek_id, a.job_id, 1 action
      from
        nlp.qyh_sample_geek_add a
      join
        nlp.qyh_sample_boss_chat b
      on
        a.geek_id = b.geek_id and a.boss_id = b.boss_id
    ) a
    group by
      a.geek_id, a.job_id, action;

  select '======================= selecting add without chat =============================' from dw_bosszp.zp_geek limit 1;

  create temporary table nlp.qyh_sample_geek_add_no_chat as
    select 
      geek_id, job_id, boss_id
    from (
      select
        a.geek_id, a.job_id, a.boss_id, b.action
      from
        nlp.qyh_sample_geek_add a
      left join
        nlp.qyh_sample_boss_chat b
      on
        a.geek_id = b.geek_id and a.boss_id = b.boss_id
    ) c
    where
      action is null
    group by
      geek_id, job_id, boss_id;

  select '======================= selecting active =============================' from dw_bosszp.zp_geek limit 1;

  create temporary table nlp.qyh_sample_active_boss as
    select 
      uid boss_id
    from
      dw_bosszp.bg_action
    where
      ds > '$start_time' and ds <= '$end_time_plus'
    and
      action = 'list-geek'
    and
      bg = 1
    group by
      uid;

  select '======================= selecting active and no chat =============================' from dw_bosszp.zp_geek limit 1;

  create temporary table nlp.qyh_sample_geek_add_no_chat_active as
    select
      a.geek_id, a.job_id, 0 action
    from
      nlp.qyh_sample_geek_add_no_chat a
    join
      nlp.qyh_sample_active_boss b
    on
      a.boss_id = b.boss_id;

  select '======================= joining samples =============================' from dw_bosszp.zp_geek limit 1;

  create temporary table nlp.qyh_sample as
    select 
      geek_id, job_id, action 
    from
    (
      select 
        geek_id, job_id, action
        from 
          nlp.qyh_sample_geek_add_boss_chat
        distribute by rand() 
        sort by rand()
        limit $n_positive
      union all
      select
        geek_id, job_id, action
        from 
          nlp.qyh_sample_geek_add_no_chat_active
        distribute by rand() 
        sort by rand()
        limit $n_negative
    ) a
    group by
      geek_id, job_id, action;

  insert overwrite table nlp.qyh_sample
    select
      geek_id, job_id, sum(action) action
    from
      nlp.qyh_sample
    group by
      geek_id, job_id;

  select '======================= joining profile =============================' from dw_bosszp.zp_geek limit 1;

  create temporary table nlp.qyh_sample_with_profile as
  select
    a.geek_id, b.cv geek_desc, a.job_id, c.jd job_desc
  from
    nlp.qyh_sample a
  join
    nlp.geek_cv_token b
  on 
    a.geek_id = b.geek_id
  join
    nlp.job_desc_token c
  on
    a.job_id = c.job_id;

  select '======================= writing files =============================' from dw_bosszp.zp_geek limit 1;

  create temporary table nlp.qyh_positive_sample_limit as
    select 
      job_id, job_desc, geek_id, geek_desc
    from
      nlp.qyh_sample_with_profile
    where action >= 1
    distribute by rand()
    sort by rand()
    limit $n_positive;

  create temporary table nlp.qyh_negative_sample_limit as
    select
      job_id, job_desc, geek_id, geek_desc
    from
      nlp.qyh_sample_with_profile
    where action = 0
    distribute by rand()
    sort by rand()
    limit $n_negative;

  insert overwrite local directory './negative'
    select * from nlp.qyh_positive_sample_limit;

  insert overwrite local directory './positive'
    select * from nlp.qyh_negative_sample_limit;

  insert overwrite local directory './all'
    select 
      geek_desc, job_desc
    from
      nlp.qyh_positive_sample_limit
    union all
    select
      geek_desc, job_desc
    from
      nlp.qyh_negative_sample_limit;
  "

  cat positive/* >> $fp/$fp.positive
  cat negative/* >> $fp/$fp.negative
  cat all/* >> $fp/$fp.all
}

fp=$1
n_sample=100000

fp=tmp
if [ -d "./$fp" ]; then
  rm -r $fp
fi
mkdir $fp

# select_add_no_chat

for i in {-30..-23}  
do  
select_add_no_chat $fp $i $[$n_sample/7]
done  