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

  drop table nlp.geek_cv_token;
  create table nlp.geek_cv_token as
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

  drop table nlp.expect_profile_category;
  create table nlp.expect_profile_category as
    select
      a.id expect_id, a.user_id geek_id, a.l3_name, a.city, 
      b.gender, b.degree, b.fresh_graduate, b.apply_status, b.completion
    from 
      dw_bosszp.geek_expect_info a
    join
      dw_bosszp.zp_geek b
    on
      a.user_id = b.geek_id;

  drop table nlp.expect_profile_dense;
  create table nlp.expect_profile_dense as
    select
      a.id expect_id, a.low_salary, a.high_salary, b.age, b.work_years
    from 
      dw_bosszp.geek_expect_info a
    join
      dw_bosszp.zp_geek b
    on
      a.user_id = b.geek_id;

  select '======================= selecting job profile =============================' from dw_bosszp.zp_geek limit 1;
  
  drop table nlp.job_desc_token;
  create table nlp.job_desc_token as
    select
      id job_id,
      concat_ws('\t', 
        tokenizer(
          split(
            regexp_replace(
              concat_ws(
                '\t',
                collect_list(job_desc, com_desc)
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

  drop table nlp.job_profile_category;
  create table nlp.job_profile_category as
    select
      id job_id, position, city, degree, experience, area_business_name,
      boss_id, boss_title, is_hr, stage
    from
      dw_bosszp.zp_job;

  drop table nlp.job_profile_dense;
  create table nlp.job_profile_dense as
    select
      id job_id, low_salary, high_salary
    from
      dw_bosszp.zp_job;

  select '======================= done =============================' from dw_bosszp.zp_geek limit 1;
  
